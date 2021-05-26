#include "pch.h"
#include "CudaMemoryAnalyzer.h"
#include "StaticAnalyzer.h"
#include <fstream>

void runAnalyzer(std::string const& filePath,
	std::string const& kernelName,
	std::vector<std::string> const& additionalIncludeDirs,
	std::vector<void const*> const& args,
	std::vector<uint64_t> const& templateArgs,
	std::string const& outputFile,
	ExecutionConfig const& executionConfig,
	char const* llvmIncludePath,
	bool checkOverflows)
{
	try {
		KernelContext kernelContext{ kernelName, {}, args, templateArgs, executionConfig.gridDim, executionConfig.blockDim, executionConfig.dynamicMemorySize };
		StaticAnalyzer staticAnalyzer(kernelContext);
		if (outputFile.empty()) {
			staticAnalyzer.Analyze(filePath, additionalIncludeDirs, std::cout, checkOverflows, llvmIncludePath);
		} else {
			std::ofstream os(outputFile);
			staticAnalyzer.Analyze(filePath, additionalIncludeDirs, os, checkOverflows, llvmIncludePath);
		}
	} catch (AnalyzerException& e) {
		std::cout << "AnalyzerException: " << e.what() << std::endl;
	} catch (std::exception& e) {
		std::cout << "std::exception: " << e.what() << std::endl;
	} catch (...) {
		std::cout << "Unknows exception was thrown" << std::endl;
	}
}

void checkBufferOverflows(std::string const& filePath,
	std::string const& kernelName,
	std::vector<std::string> const& additionalIncludeDirs,
	std::vector<void const*> const& args,
	std::vector<uint64_t> const& templateArgs,
	std::string const& outputFile,
	ExecutionConfig const& executionConfig,
	char const* llvmIncludePath)
{
	runAnalyzer(filePath, kernelName, additionalIncludeDirs, args, templateArgs,
		outputFile, executionConfig, llvmIncludePath, true);
}

void checkRestrictViolations(std::string const& filePath,
	std::string const& kernelName,
	std::vector<std::string> const& additionalIncludeDirs,
	std::vector<void const*> const& args,
	std::vector<uint64_t> const& templateArgs,
	std::string const& outputFile,
	ExecutionConfig const& executionConfig,
	char const* llvmIncludePath)
{
	runAnalyzer(filePath, kernelName, additionalIncludeDirs, args, templateArgs,
		outputFile, executionConfig, llvmIncludePath, false);
}