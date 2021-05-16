#pragma once
#include "pch.h"
#include "RestrictViolationRule.h"
#include "Solver.h"
#include "AnalyzerException.h"

RestrictViolationRule::RestrictViolationRule(RuleContext* ruleContext, ASTWalker* walker)
	: AbstractRule(ruleContext)
	, walker(walker)
{}

std::string varNameWithoutVersion(std::string const& s) {
	return s.substr(0, s.find("!"));
}

void RestrictViolationRule::apply(Statement const* kernelBody) {
	applyRule(kernelBody);
	
	std::vector<std::pair<int, int>> checkPairs;
	std::vector<z3::expr> rules;
	auto const& astContext = walker->getASTContext();
	auto const& sourceManager = walker->getSourceManager();
	for (int i = 0; i < memoryAccesses.size(); i++) {
		if (memoryAccesses[i].type == MemoryAccess::Type::READ) {
			continue;
		}
		auto const& write = memoryAccesses[i];
		std::string writePtrName = write.base->getVarName();
		bool isWritePtrRestrict = write.base->getType().type.isRestrictQualified();
		for (int j = 0; j < memoryAccesses.size(); j++) {
			auto const& other = memoryAccesses[j];
			std::string otherPtrName = other.base->getVarName();
			bool isOtherPtrRestrict = other.base->getType().type.isRestrictQualified();
			if (isWritePtrRestrict && otherPtrName != writePtrName ||
				!isWritePtrRestrict && isOtherPtrRestrict)
			{
				checkPairs.emplace_back(i, j);
				z3::expr writeAddr = getBufAddress(write.base) + write.index * 
					state->z3_ctx->int_val(elemTypeSize(write.base, astContext));
				z3::expr otherAddr = getBufAddress(other.base) + other.index *
					state->z3_ctx->int_val(elemTypeSize(other.base, astContext));
				z3::expr curRule = (writeAddr == otherAddr);
				for (auto const& predicate : memoryAccesses[j].predicates) {
					curRule = curRule && predicate;
				}
				for (auto const& predicate : memoryAccesses[i].predicates) {
					curRule = curRule && predicate;
				}
				rules.push_back(curRule);
			}
		}
	}
	if (checkPairs.size() == 0) {
		return;
	}
	z3::expr rule = rules[0];
	for (int i = 1; i < rules.size(); i++) {
		rule = rule || rules[i];
	}
	auto model = getErrorModel(state, rule);
	if (model) { 
		for (int i = 0; i < rules.size(); i++) {
			if (model->eval(rules[i]).is_true()) {
				auto write = memoryAccesses[checkPairs[i].first];
				auto other = memoryAccesses[checkPairs[i].second];
				z3::expr writeAddr = getBufAddress(write.base) + write.index * 
					state->z3_ctx->int_val(elemTypeSize(write.base, astContext));
				z3::expr otherAddr = getBufAddress(other.base) + other.index *
					state->z3_ctx->int_val(elemTypeSize(other.base, astContext));
				//std::cout << model->eval(writeAddr).to_string() << std::endl;

				//std::cout << model->eval(otherAddr).to_string() << std::endl;
				std::ostringstream message;
				std::string location = other.base->getLocation().printToString(sourceManager);
				std::filesystem::path p = location;
				location = p.filename().string();
				message << "__restrict__ semantics violation: write to " <<
					//write.base->getLocation().printToString(sourceManager) << ' ' <<
					varNameWithoutVersion(write.base->toZ3Expr(state).to_string()) << '[' <<
					model->eval(write.index).to_string() << ']' << " aliases with access to " <<
					varNameWithoutVersion(other.base->toZ3Expr(state).to_string()) << '[' <<
					model->eval(other.index).to_string() << ']' << " at " << location;
				errors.insert({ write.base->getLocation(), message.str() });
			}
		}
	}
}

z3::expr RestrictViolationRule::getBufAddress(Expression const* base) {
	if (auto var = dynamic_cast<AtomicVariable const*>(base)) {
		if (base->getType().array_size != -1) {
			return var->z3AddressExpr(state);
		}
		return var->toZ3Expr(state);
	}
	throw AnalyzerException("Unexpected");
}

uint64_t RestrictViolationRule::elemTypeSize(Expression const* base, clang::ASTContext const& astContext) {
	auto type = base->getType().type;
	if (type->isPointerType()) {
		return astContext.getTypeSize(type->getPointeeType()) / 8;
	}
	if (type->isArrayType()) {
		return astContext.getTypeSize(type->getArrayElementTypeNoTypeQual()) / 8;
	}
	throw AnalyzerException("Unknown array structure");
}

void RestrictViolationRule::handleMemoryAccesses(Expression const* expr) {
	expr->rememberMemoryAccesses(state, memoryAccesses);
}