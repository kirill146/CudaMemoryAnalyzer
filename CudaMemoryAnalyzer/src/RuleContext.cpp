#include "pch.h"
#include "RuleContext.h"

void RuleContext::rememberMemoryAccess(MemoryAccess const& memoryAccess) {
	memoryAccesses.push_back(memoryAccess);
}

uint64_t RuleContext::allocateMemory(uint64_t size) {
	uint64_t result = nextAddress;
	nextAddress += size;
	return result;
}
