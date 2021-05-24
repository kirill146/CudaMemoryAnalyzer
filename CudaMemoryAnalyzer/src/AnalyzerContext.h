#pragma once
#include "Statement.h"
#include "RuleContext.h"
#include "KernelContext.h"

struct AnalyzerContext {
	AnalyzerContext(KernelContext kernelContext)
		: kernelContext(std::move(kernelContext))
		, ruleContext(kernelContext.gridDim, kernelContext.blockDim)
	{}
	KernelContext kernelContext;
	RuleContext ruleContext;
	std::unique_ptr<Statement> kernelBody = nullptr;
};