#pragma once
#include "State.h"
#include "Function.h"

struct RecordSortDescriptor {
	z3::sort sort;
	std::unordered_map<std::string, z3::func_decl> getters;
};

class RuleContext {
public:
	RuleContext(dim3 const& gridDim, dim3 const& blockDim)
		: state(this, gridDim, blockDim)
	{}
	int acquireNextVersion(std::string const& varName) {
		if (varVersions.count(varName)) {
			return ++varVersions[varName];
		}
		return varVersions[varName] = 0;
	}
	void rememberMemoryAccess(MemoryAccess const& memoryAccess);
	uint64_t allocateMemory(uint64_t size);
private:
	std::unordered_map<std::string, int> varVersions;
	uint64_t nextAddress = 1024;
public:
	State state;
	std::unordered_map<std::string, Function> functions;
	std::vector<MemoryAccess> memoryAccesses;
	std::unordered_map<std::string, RecordSortDescriptor> recordSorts;
};