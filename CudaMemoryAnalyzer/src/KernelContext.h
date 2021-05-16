#pragma once
#include <vector>
#include <unordered_map>
#include <vector_types.h>

struct KernelContext {
	std::string kernelName;
	std::unordered_map<std::string, size_t> argSizes;
	std::vector<void const*> const scalarArgValues;
	dim3 gridDim;
	dim3 blockDim;
};