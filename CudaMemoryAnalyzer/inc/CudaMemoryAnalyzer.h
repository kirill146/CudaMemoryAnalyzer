#pragma once

#include <unordered_map>
#include <ExecutionConfig.h>

struct dim3;

void checkBufferOverflows(std::string const& filePath,
	std::string const& kernelName,
	std::vector<std::string> const& additionalIncludeDirs,
	std::vector<void const*> const& args,
	std::string const& outputFile,
	ExecutionConfig const& executionConfig,
	char const* llvmIncludePath = "");

void checkRestrictViolations(std::string const& filePath,
	std::string const& kernelName,
	std::vector<std::string> const& additionalIncludeDirs,
	std::vector<void const*> const& args,
	std::string const& outputFile,
	ExecutionConfig const& executionConfig,
	char const* llvmIncludePath = "");