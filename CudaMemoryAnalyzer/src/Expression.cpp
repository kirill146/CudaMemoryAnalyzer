#include "pch.h"
#include "Expression.h"
#include "AnalyzerException.h"
#include "RuleContext.h"

IntegerConst::IntegerConst(uint64_t val, ExpressionType const& type,
	clang::SourceLocation const& location)
	: Expression(type, location)
	, val(val)
{}

z3::expr IntegerConst::toZ3Expr(State const* state) const {
	return state->z3_ctx->int_val(val);
}

ArraySubscriptExpression::ArraySubscriptExpression(std::unique_ptr<Expression> base,
	std::unique_ptr<Expression> index, int array_size, ExpressionType const& type,
	clang::SourceLocation const& location)
	: Expression(type, location)
	, base(std::move(base))
	, index(std::move(index))
	, array_size(array_size)
{} 

z3::expr ArraySubscriptExpression::toZ3Expr(State const* state) const {
	z3::expr z3_base = base->toZ3Expr(state);
	z3::expr z3_index = index->toZ3Expr(state);
	if (base->getType().sort.is_array()) {
		return z3::select(z3_base, z3_index);
	}
	return state->getUndefined(type.sort);
}

void ArraySubscriptExpression::rememberMemoryAccesses(State* state, std::vector<MemoryAccess>& memoryAccesses) const {
	index->rememberMemoryAccesses(state, memoryAccesses);
	state->rememberMemoryAccess(base.get(), index->toZ3Expr(state), MemoryAccess::Type::READ, memoryAccesses);
}

std::string ArraySubscriptExpression::getVarName() const {
	return base->getVarName();
}

Variable::Variable(std::string name, ExpressionType const& type,
	clang::SourceLocation const& location)
	: Expression(type, location)
	, name(std::move(name))
{}

std::string Variable::getVersionedName(int version) const {
	return name + "!" + std::to_string(version);
}

z3::expr Variable::toZ3Expr(State const* state) const {
	return state->z3_ctx->constant(
		getVersionedName(state->getVariableVersion(name)).c_str(),
		type.sort);
}

AtomicVariable::AtomicVariable(std::string name, ExpressionType const& type, uint64_t address,
	clang::SourceLocation const& location)
	: Variable(name, type, location)
	, address(address)
{}

z3::expr AtomicVariable::z3AddressExpr(State const* state) const {
	std::string z3Name = "&" + getVersionedName(state->getVariableVersion(name));
	return state->z3_ctx->constant(z3Name.c_str(), state->z3_ctx->int_sort());
}

InitListExpression::InitListExpression(std::vector<std::unique_ptr<Expression>> initList,
	ExpressionType const& type, clang::SourceLocation const& location)
	: Expression(type, location)
	, initList(std::move(initList))
{}

z3::expr InitListExpression::toZ3Expr(State const* state) const {
	z3::expr result = state->getTmp(type.sort);
	for (int i = 0; i < initList.size(); i++) {
		z3::expr val = initList[i]->toZ3Expr(state);
		if (initList[i]->getType().array_size == -1) {
			result = z3::store(result, i, val);
		} else {
			throw AnalyzerException("Not implemented");	
		}
	}
	return result;
}

CallExpression::CallExpression(std::string functionName,
	std::vector<std::unique_ptr<Expression>> arguments, ExpressionType const& type,
	clang::SourceLocation const& location)
	: Expression(type, location)
	, functionName(std::move(functionName))
	, arguments(std::move(arguments))
{}

z3::expr CallExpression::toZ3Expr(State const* state) const {
	return state->getUndefined(type.sort);
}

RealConst::RealConst(double val, ExpressionType const& type, clang::SourceLocation const& location)
	: Expression(type, location)
	, val(val)
{}

z3::expr RealConst::toZ3Expr(State const* state) const {
	return state->z3_ctx->real_val(std::to_string(val).c_str());
}

BoolConst::BoolConst(bool val, ExpressionType const& type, clang::SourceLocation const& location)
	: Expression(type, location)
	, val(val)
{}

z3::expr BoolConst::toZ3Expr(State const* state) const {
	return state->z3_ctx->bool_val(val);
}

PodStruct::PodStruct(std::string name, std::vector<std::unique_ptr<Variable>> fields,
	ExpressionType const& type, clang::SourceLocation const& location)
	: Variable(name, type, location)
	, fields(std::move(fields))
{}

z3::expr PodStruct::toZ3Expr(State const* state) const {
	state;
	throw AnalyzerException("PosStruct::toZ3Expr() is undefined");
}

ImplicitCastExpression::ImplicitCastExpression(std::unique_ptr<Expression> subExpr, clang::CastKind kind, ExpressionType const& type, clang::SourceLocation const& location)
	: Expression(type, location)
	, subExpr(std::move(subExpr))
	, kind(kind)
{
}

z3::expr ImplicitCastExpression::toZ3Expr(State const* state) const {
	if (kind == clang::CastKind::CK_ArrayToPointerDecay) {
		if (!subExpr->getType().sort.is_array()) {
			throw AnalyzerException("ImplicitCastExpression: unexpected ArrayToPointerDecay cast");
		}
		auto* arr = dynamic_cast<AtomicVariable const*>(subExpr.get());
		if (!arr) {
			throw AnalyzerException("ImplicitCastExpression: invalid dynamic cast");
		}
		return arr->z3AddressExpr(state);
	}
	return subExpr->toZ3Expr(state);
}

MemberExpression::MemberExpression(std::unique_ptr<Expression> base, std::string recordName,
	std::string memberName, ExpressionType const& type, clang::SourceLocation const& location)
	: Expression(type, location)
	, base(std::move(base))
	, recordName(std::move(recordName))
	, memberName(std::move(memberName))
{}

z3::expr MemberExpression::toZ3Expr(State const* state, z3::expr const& recordVar, std::string const& fieldName) const {
	/*z3::expr varExpr = state->z3_ctx->constant(
		var->getVersionedName(varVersion).c_str(),
		var->getType().sort); */
	return state->ruleContext->recordSorts.at(recordName).getters.at(fieldName)(recordVar);
}

z3::expr MemberExpression::toZ3Expr(State const* state) const {
	z3::expr baseExpr = base->toZ3Expr(state);
	return state->ruleContext->recordSorts.at(recordName).getters.at(memberName)(baseExpr);
	//return toZ3Expr(state, state->getVariableVersion(var->getName()), memberName);
}