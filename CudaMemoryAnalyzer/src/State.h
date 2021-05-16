#pragma once
#include <z3++.h>
#include <unordered_set>
#include "ExpressionType.h"
#include "MemoryAccess.h"
#include <optional>

class RuleContext;
struct dim3;

class State {
public:
	explicit State(RuleContext* ruleContext, dim3 const& gridDim, dim3 const& blockDim);
	explicit State(State* prevState);
	int getVariableVersion(std::string const& name) const;
	std::vector<z3::expr const*> getPredicates() const;
	std::unordered_set<std::string> getVisibleVariables() const;
	std::optional<ExpressionType> getVariableType(std::string const& name) const;
	z3::expr getTmp(z3::sort const& sort) const;
	z3::expr getUndefined(z3::sort const& sort) const;
	int acquireNextVersion(std::string const& name);
	void rememberMemoryAccess(Expression const* base, z3::expr const& index,
		MemoryAccess::Type type, std::vector<MemoryAccess>& memoryAccesses);
	z3::expr getExpr(std::string const& name);
	//uint64_t acqureAddress(uint64_t size);
public:
	RuleContext* ruleContext;
	std::shared_ptr<z3::context> z3_ctx = std::make_shared<z3::context>();
	std::unordered_map<std::string, int> variableVersions;
	std::unordered_map<std::string, ExpressionType> variableTypes;
	std::unordered_map<std::string, int> localArraySizes;
	std::vector<z3::expr> localVariablesPredicates;
	std::vector<z3::expr> predicates;
	int mutable tmpVersion = 0;
	State* prevState = nullptr;
};