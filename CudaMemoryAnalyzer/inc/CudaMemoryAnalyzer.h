#pragma once
#include <string>
#include <vector>
#include <ExecutionConfig.h>

void checkBufferOverflows(std::string const& filePath,
	std::string const& kernelName,
	std::vector<std::string> const& additionalIncludeDirs,
	std::vector<void const*> const& args,
	std::vector<uint64_t> const& templateArgs,
	std::string const& outputFile,
	ExecutionConfig const& executionConfig,
	char const* llvmIncludePath = "");

void checkRestrictViolations(std::string const& filePath,
	std::string const& kernelName,
	std::vector<std::string> const& additionalIncludeDirs,
	std::vector<void const*> const& args,
	std::vector<uint64_t> const& templateArgs,
	std::string const& outputFile,
	ExecutionConfig const& executionConfig,
	char const* llvmIncludePath = "");