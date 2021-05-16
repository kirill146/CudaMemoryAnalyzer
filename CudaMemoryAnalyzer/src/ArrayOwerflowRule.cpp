#include "pch.h"
#include "ArrayOwerflowRule.h"
#include <z3++.h>
#include "Solver.h"

ArrayOverflowRule::ArrayOverflowRule(RuleContext* ruleContext)
	: AbstractRule(ruleContext)
{}

void ArrayOverflowRule::visitArraySubscriptExpression(ArraySubscriptExpression const* arraySubscriptExpression, bool needCheck) {
	if (needCheck) {
		z3::expr index = arraySubscriptExpression->getIndex()->toZ3Expr(state);
		int array_size = arraySubscriptExpression->getArraySize();
		if (auto varBase = dynamic_cast<AtomicVariable const*>(arraySubscriptExpression->getBase())) {
			if (state->localArraySizes.count(varBase->getName())) {
				array_size = state->localArraySizes[varBase->getName()];
			}
		}
		if (array_size != -1) {
			auto model = getErrorModel(state, index >= array_size || index < 0);
			if (model) {
				std::ostringstream message;
				message << "Array with size " << array_size << " accessed at index " << model->eval(index);
				//std::cout << "z3 index expression: " << index.to_string() << std::endl;
				errors.insert({ arraySubscriptExpression->getLocation(), message.str() });
			}
		}
	}
	AbstractRule::visitArraySubscriptExpression(arraySubscriptExpression, needCheck);
}
