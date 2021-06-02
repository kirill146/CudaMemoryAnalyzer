#include "pch.h"
#include "AbstractRule.h"
#include "Statement.h"
#include "Solver.h"
#include "AnalyzerException.h"

AbstractRule::AbstractRule(RuleContext* ruleContext)
	: state(&ruleContext->state)
{}

void AbstractRule::apply(Statement const* kernelBody) {
	applyRule(kernelBody);
}

void AbstractRule::applyRule(Statement const* statement, bool needCheck) {
	if (statement == nullptr) {
		return;
	}
	assert(statement != nullptr);

	if (dynamic_cast<IntegerConst const*>(statement) ||
		dynamic_cast<BoolConst const*>(statement) ||
		dynamic_cast<RealConst const*>(statement) ||
		dynamic_cast<AtomicVariable const*>(statement) ||
		dynamic_cast<MemberExpression const*>(statement))
	{
		// don't apply any rules to these statements
		return;
	}
	if (auto compoundStatement = dynamic_cast<CompoundStatement const*>(statement)) {
		visitCompoundStatement(compoundStatement, needCheck);
	} else if (auto declStatement = dynamic_cast<DeclStatement const*>(statement)) {
		visitDeclStatement(declStatement, needCheck);
	} else if (auto ifStatement = dynamic_cast<IfStatement const*>(statement)) {
		visitIfStatement(ifStatement, needCheck);
	} else if (auto whileStatement = dynamic_cast<WhileStatement const*>(statement)) {
		visitWhileStatement(whileStatement, needCheck);
	} else if (auto forStatement = dynamic_cast<ForStatement const*>(statement)) {
		visitForStatement(forStatement, needCheck);
	} else if (auto arraySubscriptExpression = dynamic_cast<ArraySubscriptExpression const*>(statement)) {
		visitArraySubscriptExpression(arraySubscriptExpression, needCheck);
	} else if (auto returnStatement = dynamic_cast<ReturnStatement const*>(statement)) {
		visitReturnStatement(returnStatement, needCheck);
	} else if (auto binaryOperator = dynamic_cast<BinaryOperator const*>(statement)) {
		visitBinaryOperator(binaryOperator, needCheck);
	} else if (auto unaryOperator = dynamic_cast<UnaryOperator const*>(statement)) {
		visitUnaryOperator(unaryOperator, needCheck);
	} else if (auto conditionalOperator = dynamic_cast<ConditionalOperator const*>(statement)) {
		visitConditionalOperator(conditionalOperator, needCheck);
	} else if (auto initListExpression = dynamic_cast<InitListExpression const*>(statement)) {
		visitInitListExpressionExpression(initListExpression, needCheck);
	} else if (auto callExpression = dynamic_cast<CallExpression const*>(statement)) {
		visitCallExpression(callExpression, needCheck);
	} else if (auto implicitCastExpr = dynamic_cast<ImplicitCastExpression const*>(statement)) {
		applyRule(implicitCastExpr->getSubExpr(), needCheck);
	} else {
		std::cout << "applyRule: unknown statement\n";
		throw AnalyzerException("applyRule: unknown statement");
	}

	statement->modifyState(state);
}

void AbstractRule::visitCompoundStatement(CompoundStatement const* compoundStatement, bool needCheck) {
	for (auto& statement : compoundStatement->getStatements()) {
		if (auto expr = dynamic_cast<Expression const*>(statement.get())) {
			if (needCheck) {
				handleMemoryAccesses(expr);
			}
		}
		applyRule(statement.get(), needCheck);
	}
}

void AbstractRule::visitDeclStatement(DeclStatement const* declStatement, bool needCheck) {
	if (declStatement->getInitializer()) {
		if (needCheck) {
			handleMemoryAccesses(declStatement->getInitializer());
		}
		applyRule(declStatement->getInitializer(), needCheck);
	}
}

void AbstractRule::visitIfStatement(IfStatement const* ifStatement, bool needCheck) {
	z3::expr cond = ifStatement->getCond()->toZ3Expr(state);
	State thenState(state);
	thenState.predicates.emplace_back(cond);
	state = &thenState;
	if (needCheck) {
		handleMemoryAccesses(ifStatement->getCond());
	}
	applyRule(ifStatement->getThen(), needCheck);
	state = state->prevState;

	std::unique_ptr<State> elseState = nullptr;
	if (ifStatement->getElse()) {
		elseState = std::make_unique<State>(state);
		elseState->predicates.emplace_back(!ifStatement->getCond()->toZ3Expr(elseState.get()));
		state = elseState.get();
		applyRule(ifStatement->getElse(), needCheck);
		state = state->prevState;
	}

	// merge
	for (auto const& varName : state->getVisibleVariables()) {
		int varVersion = state->getVariableVersion(varName);
		int thenVersion = thenState.getVariableVersion(varName);
		int elseVersion = elseState ? elseState->getVariableVersion(varName) : varVersion;
		if (thenVersion == varVersion && elseVersion == varVersion) {
			continue;
		}
		if (state->getVariableVersion(varName) == -1) {
			continue; // local var
		}
		ExpressionType type = *state->getVariableType(varName);
		AtomicVariable var(varName, type, UNUSED_ADDR, clang::SourceLocation());
		z3::expr oldValue = var.toZ3Expr(state);
		auto oldVarAddrExpr = var.z3AddressExpr(state);
		state->variableVersions[varName] = state->acquireNextVersion(varName);
		state->localVariablesPredicates.emplace_back(
			var.z3AddressExpr(state) == oldVarAddrExpr
		);
		z3::expr newValue = var.toZ3Expr(state);
		z3::expr thenValue = var.toZ3Expr(&thenState);
		z3::expr elseValue = elseState ? var.toZ3Expr(elseState.get()) : oldValue;
		state->localVariablesPredicates.emplace_back(
			z3::ite(cond, newValue == thenValue, newValue == elseValue));
		state->localVariablesPredicates.emplace_back(
			z3::ite(cond, newValue == thenValue, newValue == elseValue));
	}

	for (auto& localPredicate : thenState.localVariablesPredicates) {
		state->localVariablesPredicates.push_back(localPredicate);
	}
	if (elseState) {
		for (auto& localPredicate : elseState->localVariablesPredicates) {
			state->localVariablesPredicates.push_back(localPredicate);
		}
	}
}

void AbstractRule::visitWhileStatement(WhileStatement const* whileStatement, bool needCheck) {
	int checkStep = 1;
	int iter = 0;
	{ 
		State newState(state);
		state = &newState;
		for (; iter <= LOOP_ITERATIONS_BOUND; iter++) {
			z3::expr cond = whileStatement->getCond()->toZ3Expr(state);
			applyRule(whileStatement->getCond(), false);
			state->predicates.emplace_back(cond);
			if (iter == checkStep) {
				checkStep *= 10;
				if (!isReachable(&newState)) {
					break;
				}
			}
			applyRule(whileStatement->getBody(), false);
		}
		state = state->prevState;
	}

	int l = 0, r = iter + 1;
	while (r - l > 1) {
		State newState(state);
		state = &newState;
		int m = (l + r) / 2;
		for (int i = 0; i < m; i++) {
			z3::expr cond = whileStatement->getCond()->toZ3Expr(state);
			applyRule(whileStatement->getCond(), false);
			state->predicates.emplace_back(cond);
			applyRule(whileStatement->getBody(), false);
		}
		if (!isReachable(state)) {
			r = m;
		} else {
			l = m;
		}
		state = state->prevState;
	}
	int maxIterationsCount = l;
	
	{
		checkStep = maxIterationsCount / LOOP_CHECKS_COUNT + 1;
		State newState(state);
		state = &newState;
		//std::cout << "max iters: " << maxIterationsCount << std::endl;
		for (int i = 0; i < maxIterationsCount; i++) {
			z3::expr cond = whileStatement->getCond()->toZ3Expr(state);
			bool doCheck = (i % checkStep == 0) || (i == maxIterationsCount - 1)
				? needCheck : false;
			if (needCheck) {
				handleMemoryAccesses(whileStatement->getCond());
			}
			applyRule(whileStatement->getCond(), doCheck);
			state->predicates.emplace_back(cond);
			applyRule(whileStatement->getBody(), doCheck);
		}
		if (needCheck) {
			handleMemoryAccesses(whileStatement->getCond());
		}
		applyRule(whileStatement->getCond(), needCheck);
		state = state->prevState;

		for (auto const& varName : state->getVisibleVariables()) {
			/*int varVersion = state->getVariableVersion(varName);
			int lastIterVersion = newState.getVariableVersion(varName);
			if (lastIterVersion == varVersion) {
				continue;
			}*/
			state->variableVersions[varName] = newState.getVariableVersion(varName);
			/*ExpressionType type = *state->getVariableType(varName);
			state->variableVersions[varName] = state->acquireNextVersion(varName);
			AtomicVariable var(varName, type, UNUSED_ADDR, clang::SourceLocation());
			z3::expr newValue = var.toZ3Expr(state);
			z3::expr lastIterValue = var.toZ3Expr(&newState);
			state->localVariablesPredicates.emplace_back(newValue == lastIterValue);*/
		}
		for (auto localPredicate : newState.localVariablesPredicates) {
			state->localVariablesPredicates.push_back(localPredicate);
		}
	}
}

void AbstractRule::visitForStatement(ForStatement const* forStatement, bool needCheck) {
	applyRule(forStatement->getInit(), needCheck);
	applyRule(forStatement->getTransformed(), needCheck);
}

void AbstractRule::visitArraySubscriptExpression(ArraySubscriptExpression const* arraySubscriptExpression, bool needCheck) {
	applyRule(arraySubscriptExpression->getBase(), needCheck);
	applyRule(arraySubscriptExpression->getIndex(), needCheck);
}

void AbstractRule::visitInitListExpressionExpression(InitListExpression const* initListExpression, bool needCheck) {
	for (auto& e : initListExpression->getInitList()) {
		applyRule(e.get(), needCheck);
	}
}

void AbstractRule::visitReturnStatement(ReturnStatement const* returnStatement, bool needCheck) {
	if (needCheck) {
		handleMemoryAccesses(returnStatement->getExpr());
	}
	applyRule(returnStatement->getExpr(), needCheck);
}

void AbstractRule::visitBinaryOperator(BinaryOperator const* binaryOperator, bool needCheck) {
	applyRule(binaryOperator->getLhs(), needCheck);
	applyRule(binaryOperator->getRhs(), needCheck);
}

void AbstractRule::visitUnaryOperator(UnaryOperator const* unaryOperator, bool needCheck) {
	applyRule(unaryOperator->getArg(), needCheck);
}

void AbstractRule::visitConditionalOperator(ConditionalOperator const* conditionalOperator, bool needCheck) {
	applyRule(conditionalOperator->getCond(), needCheck);
	applyRule(conditionalOperator->getThen(), needCheck);
	applyRule(conditionalOperator->getElse(), needCheck);
}

void AbstractRule::visitCallExpression(CallExpression const* callExpression, bool needCheck) {
	for (auto& arg : callExpression->getArguments()) {
		applyRule(arg.get(), needCheck);
	}
	if (state->ruleContext->functions.count(callExpression->getFunctionName())) {
		Function const& function = state->ruleContext->functions.at(callExpression->getFunctionName());
		State newState(state);
		state = &newState;
		for (int i = 0; i < function.getArgs().size(); i++) {
			std::unique_ptr<AtomicVariable> const& variable = function.getArgs()[i];
			std::unique_ptr<Expression> const& arg = callExpression->getArguments()[i];
			z3::expr value = arg->toZ3Expr(state->prevState);
			std::string name = variable->getName();
			state->localArraySizes[name] = arg->getType().array_size;
			state->variableVersions[name] = state->acquireNextVersion(name);
			state->variableTypes.insert({ name, { arg->getType().type, arg->getType().sort, arg->getType().array_size } });
			uint64_t size = std::max(arg->getType().array_size, 10) * 8;
		
			AtomicVariable newVariable(name, arg->getType(), state->ruleContext->allocateMemory(size), variable->getLocation());
			state->localVariablesPredicates.push_back(z3::expr(newVariable.toZ3Expr(state) == value));
			state->localVariablesPredicates.emplace_back(
				newVariable.z3AddressExpr(state) == state->z3_ctx->int_val(newVariable.getAddress()));
		}
		applyRule(function.getBody(), needCheck);
		state = newState.prevState;
	}
}
