#pragma once

#include <unordered_map>

struct dim3;

void checkBufferOverflows(std::string const& filePath,
	std::string const& kernelName,
	std::vector<std::string> const& additionalIncludeDirs,
	std::vector<void const*> const& args,
	std::string const& outputFile,
	dim3 const& gridDim,
	dim3 const& blockDim,
	char const* llvmIncludePath = "");

void checkRestrictViolations(std::string const& filePath,
	std::string const& kernelName,
	std::vector<std::string> const& additionalIncludeDirs,
	std::vector<void const*> const& args,
	std::string const& outputFile,
	dim3 const& gridDim,
	dim3 const& blockDim,
	char const* llvmIncludePath = "");