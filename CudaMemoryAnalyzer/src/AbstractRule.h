#pragma once
#include "Expression.h"
#include "RuleContext.h"
#include "Operator.h"

class AbstractRule {
public:
	AbstractRule(RuleContext* ruleContext);
	virtual ~AbstractRule() = default;
	virtual void apply(Statement const* kernelBody);
protected:
	void applyRule(Statement const* statement, bool needCheck = true);
	virtual void visitCompoundStatement(CompoundStatement const* compoundStatement, bool needCheck);
	virtual void visitDeclStatement(DeclStatement const* declStatement, bool needCheck);
	virtual void visitIfStatement(IfStatement const* ifStatement, bool needCheck);
	virtual void visitWhileStatement(WhileStatement const* whileStatement, bool needCheck);
	virtual void visitForStatement(ForStatement const* forStatement, bool needCheck);
	virtual void visitArraySubscriptExpression(ArraySubscriptExpression const* arraySubscriptExpression, bool needCheck);
	virtual void visitInitListExpressionExpression(InitListExpression const* initListExpression, bool needCheck);
	virtual void visitReturnStatement(ReturnStatement const* returnStatement, bool needCheck);
	virtual void visitBinaryOperator(BinaryOperator const* binaryOperator, bool needCheck);
	virtual void visitUnaryOperator(UnaryOperator const* unaryOperator, bool needCheck);
	virtual void visitConditionalOperator(ConditionalOperator const* conditionalOperator, bool needCheck);
	virtual void visitCallExpression(CallExpression const* callExpression, bool needCheck);
	virtual void handleMemoryAccesses(Expression const* expr) { expr; }
public:
	std::map<clang::SourceLocation, std::string> errors;
protected:
	State* state;
	static int constexpr LOOP_ITERATIONS_BOUND = 1000000;
	static int constexpr LOOP_CHECKS_COUNT = 20;
};