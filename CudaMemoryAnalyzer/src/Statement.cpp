#include "pch.h"
#include "Statement.h"
#include "Expression.h"
#include "AnalyzerException.h"

ReturnStatement::ReturnStatement(std::unique_ptr<Expression> expr,
	clang::SourceLocation const& location)
	: Statement(location)
	, expr(std::move(expr))
{}

CompoundStatement::CompoundStatement(std::vector<std::unique_ptr<Statement>> statements,
	clang::SourceLocation const& location)
	: Statement(location)
	, statements(std::move(statements))
{}

DeclStatement::DeclStatement(std::unique_ptr<AtomicVariable> variable, std::unique_ptr<Expression> initializer,
	clang::SourceLocation const& location)
	: Statement(location)
	, variable(std::move(variable))
	, initializer(std::move(initializer))
{}

void DeclStatement::modifyState(State* state) const {
	if (variable == nullptr ||
		variable->getType().sort.name().str() == "unknown_sort")
	{
		PEDANTIC_THROW("Cannot assign value");
		return;
	}
	std::string varName = variable->getName();
	state->variableVersions[varName] = state->acquireNextVersion(varName);
	state->variableTypes.insert({ varName, variable->getType() });
	if (initializer) {
		if (initializer->getType().sort.name().str() == "unknown_sort") {
			PEDANTIC_THROW("Cannot assign value");
			return;
		}
		if (variable->getType().sort.is_int() && initializer->getType().sort.is_array()) {
			auto* arr = dynamic_cast<AtomicVariable const*>(initializer.get());
			if (!arr) {
				throw AnalyzerException("DeclStatement: invalid dynamic cast");
			}
			state->localVariablesPredicates.emplace_back(
				variable->toZ3Expr(state) == arr->z3AddressExpr(state)
			);
		} else {
			state->localVariablesPredicates.emplace_back(
				variable->toZ3Expr(state) == initializer->toZ3Expr(state)
			);
		}
	}
	if (variable->getType().array_size != -1) {
		state->localVariablesPredicates.emplace_back(
			variable->z3AddressExpr(state) == state->z3_ctx->int_val(variable->getAddress())
		);
	}
}

Expression::Expression(ExpressionType const& type, clang::SourceLocation const& location)
	: Statement(location)
	, type(type)
{}

IfStatement::IfStatement(std::unique_ptr<Expression> cond, std::unique_ptr<Statement> then,
	std::unique_ptr<Statement> _else, clang::SourceLocation const& location)
	: Statement(location)
	, cond(std::move(cond))
	, then(std::move(then))
	, _else(std::move(_else))
{}

WhileStatement::WhileStatement(std::unique_ptr<Expression> cond, std::unique_ptr<Statement> body,
	clang::SourceLocation const& location)
	: Statement(location)
	, cond(std::move(cond))
	, body(std::move(body))
{}

ForStatement::ForStatement(std::unique_ptr<Statement> init, std::unique_ptr<Expression> cond,
	std::unique_ptr<Expression> inc, std::unique_ptr<Statement> body, clang::SourceLocation const& location)
	: Statement(location)
	, init(std::move(init))
{
	std::vector<std::unique_ptr<Statement>> transformedBody;
	clang::SourceLocation body_location = body->getLocation();
	transformedBody.push_back(std::move(body));
	transformedBody.push_back(std::move(inc));
	transformed = std::make_unique<WhileStatement>(std::move(cond),
		std::make_unique<CompoundStatement>(std::move(transformedBody), body_location),
		location);
}

Statement::Statement(clang::SourceLocation location)
	: location(std::move(location))
{}
