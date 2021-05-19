#pragma once
#include "KernelContext.h"
#include "AnalyzerContext.h"
#include <filesystem>

class StaticAnalyzer {
public:
	StaticAnalyzer(KernelContext kernelContext);
	void Analyze(
		std::filesystem::path const& filePath,
		std::vector<std::string> const& additionalIncludeDirs,
		std::ostream& os,
		bool checkOverflows = true,
		char const* llvmIncludePath = "");
private:
	AnalyzerContext analyzerContext;
};