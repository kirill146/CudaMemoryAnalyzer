#include "pch.h"
#include "StaticAnalyzer.h"
#include "ASTWalker.h"
#include "ArrayOwerflowRule.h"
#include "RestrictViolationRule.h"
#include "Solver.h"

StaticAnalyzer::StaticAnalyzer(KernelContext kernelContext)
	: analyzerContext(std::move(kernelContext))
{}

void StaticAnalyzer::Analyze(
	std::filesystem::path const& filePath,
	std::vector<std::string> const& additionalIncludeDirs,
	std::ostream& os,
	bool checkOverflows,
	char const* llvmIncludePath)
{
	ASTWalker walker(&analyzerContext, additionalIncludeDirs, llvmIncludePath);
	walker.walk(filePath); 

	std::unique_ptr<AbstractRule> rule;
	if (checkOverflows) {
		rule = std::make_unique<ArrayOverflowRule>(&analyzerContext.ruleContext);
	} else {
		rule = std::make_unique<RestrictViolationRule>(&analyzerContext.ruleContext, &walker);
	}
	rule->apply(analyzerContext.kernelBody.get());

	for (auto entry : rule->errors) {
		std::string errorLocation = entry.first.printToString(walker.getSourceManager());
		errorLocation = std::filesystem::path(errorLocation).filename().string();
		os << "[Error at " << errorLocation << "] " << entry.second << std::endl;
	}
}
