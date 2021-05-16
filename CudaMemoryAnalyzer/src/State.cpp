#include "pch.h"
#include "State.h"
#include "RuleContext.h"
#include <optional>
#include <vector_types.h>

State::State(RuleContext* ruleContext, dim3 const& gridDim, dim3 const& blockDim)
	: ruleContext(ruleContext)
{
	z3::expr gridDimX   = getExpr("gridDim.x");
	z3::expr gridDimY   = getExpr("gridDim.y");
	z3::expr gridDimZ   = getExpr("gridDim.z");
	z3::expr blockDimX  = getExpr("blockDim.x");
	z3::expr blockDimY  = getExpr("blockDim.y");
	z3::expr blockDimZ  = getExpr("blockDim.z");
	z3::expr blockIdxX  = getExpr("blockIdx.x");
	z3::expr blockIdxY  = getExpr("blockIdx.y");
	z3::expr blockIdxZ  = getExpr("blockIdx.z");
	z3::expr threadIdxX = getExpr("threadIdx.x");
	z3::expr threadIdxY = getExpr("threadIdx.y");
	z3::expr threadIdxZ = getExpr("threadIdx.z");
	
	predicates.push_back(gridDimX == (int)gridDim.x);
	predicates.push_back(gridDimY == (int)gridDim.y);
	predicates.push_back(gridDimZ == (int)gridDim.z);
	
	predicates.push_back(blockDimX == (int)blockDim.x);
	predicates.push_back(blockDimY == (int)blockDim.y);
	predicates.push_back(blockDimZ == (int)blockDim.z);
	
	predicates.push_back(blockIdxX >= 0 && blockIdxX < (int)gridDim.x);
	predicates.push_back(blockIdxY >= 0 && blockIdxY < (int)gridDim.y);
	predicates.push_back(blockIdxZ >= 0 && blockIdxZ < (int)gridDim.z);
	
	predicates.push_back(threadIdxX >= 0 && threadIdxX < (int)blockDim.x);
	predicates.push_back(threadIdxY >= 0 && threadIdxY < (int)blockDim.y);
	predicates.push_back(threadIdxZ >= 0 && threadIdxZ < (int)blockDim.z);
}

State::State(State* prevState)
	: ruleContext(prevState->ruleContext)
	, prevState(prevState)
	, z3_ctx(prevState->z3_ctx)
	, tmpVersion(prevState->tmpVersion)
{}

int State::getVariableVersion(std::string const& name) const {
	if (variableVersions.count(name)) {
		return variableVersions.at(name);
	}
	if (prevState) {
		return prevState->getVariableVersion(name);
	}
	return -1;
}

std::vector<z3::expr const*> State::getPredicates() const {
	std::vector<z3::expr const*> res = prevState ?
		prevState->getPredicates() : std::vector<z3::expr const*>();
	for (auto& pred : localVariablesPredicates) {
		res.push_back(&pred);
	}
	for (auto& pred : predicates) {
		res.push_back(&pred);
	}
	return res;
}

std::unordered_set<std::string> State::getVisibleVariables() const {
	std::unordered_set<std::string> result = prevState ? prevState->getVisibleVariables()
													   : std::unordered_set<std::string>{};
	for (auto entry : variableVersions) {
		result.insert(entry.first);
	}
	return result;
}

std::optional<ExpressionType> State::getVariableType(std::string const& name) const {
	if (variableTypes.count(name)) {
		return variableTypes.at(name);
	}
	if (prevState) {
		return prevState->getVariableType(name);
	}
	return {};
}

z3::expr State::getTmp(z3::sort const& sort) const {
	std::string name = "tmp!!" + std::to_string(tmpVersion++);
	return z3_ctx->constant(name.c_str(), sort);
}

z3::expr State::getUndefined(z3::sort const& sort) const {
	std::string name = "undefined!!" + std::to_string(tmpVersion++);
	return z3_ctx->constant(name.c_str(), sort);
}

int State::acquireNextVersion(std::string const& name) {
	return ruleContext->acquireNextVersion(name);
}

void State::rememberMemoryAccess(Expression const* base, z3::expr const& index,
	MemoryAccess::Type type, std::vector<MemoryAccess>& memoryAccesses)
{
	memoryAccesses.push_back({ base, index, type, predicates });
	std::string str_type = (type == MemoryAccess::Type::WRITE ? "write" : "read");
	//std::cout << str_type << ": " << base->toZ3Expr(this).to_string() << ' ' << index.to_string() << std::endl;
}

z3::expr State::getExpr(std::string const& name) {
	std::string versionedName = name + "!-1";
	return z3_ctx->int_const(versionedName.c_str());
}
