#pragma once
#include <vector_types.h>

class ExecutionConfig {
public:
	ExecutionConfig(dim3 const& gridDim, dim3 const& blockDim, size_t dynamicMemorySize = 0)
		: gridDim(gridDim)
		, blockDim(blockDim)
		, dynamicMemorySize(dynamicMemorySize)
	{}
public:
	dim3 const gridDim;
	dim3 const blockDim;
	size_t const dynamicMemorySize;
};