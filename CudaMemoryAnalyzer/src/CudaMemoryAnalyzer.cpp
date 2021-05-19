#include "pch.h"
#include "CudaMemoryAnalyzer.h"
#include "StaticAnalyzer.h"
#include <fstream>

void checkBufferOverflows(std::string const& filePath,
	std::string const& kernelName,
	std::vector<std::string> const& additionalIncludeDirs,
	std::vector<void const*> const& args,
	std::string const& outputFile,
	dim3 const& gridDim,
	dim3 const& blockDim,
	char const* llvmIncludePath)
{
	KernelContext kernelContext{ kernelName, {}, args, gridDim, blockDim };
	StaticAnalyzer staticAnalyzer(kernelContext);
	if (outputFile.empty()) {
		staticAnalyzer.Analyze(filePath, additionalIncludeDirs, std::cout, true, llvmIncludePath);
	} else {
		std::ofstream os(outputFile);
		staticAnalyzer.Analyze(filePath, additionalIncludeDirs, os, true, llvmIncludePath);
	}
}

void checkRestrictViolations(std::string const& filePath,
	std::string const& kernelName,
	std::vector<std::string> const& additionalIncludeDirs,
	std::vector<void const*> const& args,
	std::string const& outputFile,
	dim3 const& gridDim,
	dim3 const& blockDim,
	char const* llvmIncludePath)
{
	KernelContext kernelContext{ kernelName, {}, args, gridDim, blockDim };
	StaticAnalyzer staticAnalyzer(kernelContext);
	if (outputFile.empty()) {
		staticAnalyzer.Analyze(filePath, additionalIncludeDirs, std::cout, false, llvmIncludePath);
	} else {
		std::ofstream os(outputFile);
		staticAnalyzer.Analyze(filePath, additionalIncludeDirs, os, false, llvmIncludePath);
	}
}