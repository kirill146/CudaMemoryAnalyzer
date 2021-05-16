#pragma once
#include "AbstractRule.h"

class ArrayOverflowRule : public AbstractRule {
public:
	ArrayOverflowRule(RuleContext* ruleContext);
protected:
	void visitArraySubscriptExpression(ArraySubscriptExpression const* arraySubscriptExpression, bool needCheck) override;
};