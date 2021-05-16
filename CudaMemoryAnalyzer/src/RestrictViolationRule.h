#pragma once
#include "AbstractRule.h"
#include "ASTWalker.h"

class RestrictViolationRule : public AbstractRule {
public:
	RestrictViolationRule(RuleContext* ruleContext, ASTWalker* walker);
	void apply(Statement const* kernelBody) override;
private:
	z3::expr getBufAddress(Expression const* base);
	uint64_t elemTypeSize(Expression const* base, clang::ASTContext const& astContext);
protected:
	void handleMemoryAccesses(Expression const* expr) override;
private:
	std::vector<MemoryAccess> memoryAccesses;
	ASTWalker* walker;
};