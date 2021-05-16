#pragma once
#include "State.h"
#include "ExpressionType.h"

class Expression;
class AtomicVariable;

class Statement {
public:
	Statement(clang::SourceLocation location);
	virtual ~Statement() = default;
	virtual void modifyState(State*) const {}
	clang::SourceLocation getLocation() const { return location; }
private:
	clang::SourceLocation location;
};

class DeclStatement : public Statement {
public:
	DeclStatement(std::unique_ptr<AtomicVariable> variable, std::unique_ptr<Expression> initializer,
		clang::SourceLocation const& location);
	AtomicVariable const* getVariable() const { return variable.get(); }
	Expression const* getInitializer() const { return initializer.get(); }
	void modifyState(State* state) const override;
private:
	std::unique_ptr<AtomicVariable> variable;
	std::unique_ptr<Expression> initializer;
};

class CompoundStatement : public Statement {
public:
	CompoundStatement(std::vector<std::unique_ptr<Statement>> statements,
		clang::SourceLocation const& location);
	std::vector<std::unique_ptr<Statement>> const& getStatements() const { return statements; }
private:
	std::vector<std::unique_ptr<Statement>> statements;
};

class ReturnStatement : public Statement {
public:
	ReturnStatement(std::unique_ptr<Expression> expr, clang::SourceLocation const& location);
	Expression const* getExpr() const { return expr.get(); }
private:
	std::unique_ptr<Expression> expr;
};

class IfStatement : public Statement {
public:
	IfStatement(std::unique_ptr<Expression> cond, std::unique_ptr<Statement> then,
		std::unique_ptr<Statement> _else, clang::SourceLocation const& location);
	Expression const* getCond() const { return cond.get(); }
	Statement const* getThen() const { return then.get(); }
	Statement const* getElse() const { return _else.get(); }
private:
	std::unique_ptr<Expression> cond;
	std::unique_ptr<Statement> then;
	std::unique_ptr<Statement> _else;
};

class WhileStatement : public Statement {
public:
	WhileStatement(std::unique_ptr<Expression> cond, std::unique_ptr<Statement> body,
		clang::SourceLocation const& location);
	Expression const* getCond() const { return cond.get(); }
	Statement const* getBody() const { return body.get(); }
private:
	std::unique_ptr<Expression> cond;
	std::unique_ptr<Statement> body;
};

class ForStatement : public Statement {
public:
	ForStatement(std::unique_ptr<Statement> init, std::unique_ptr<Expression> cond,
		std::unique_ptr<Expression> inc, std::unique_ptr<Statement> body,
		clang::SourceLocation const& location);
	Statement const* getInit() const { return init.get(); }
	Statement const* getTransformed() const { return transformed.get(); }
private:
	std::unique_ptr<Statement> init;
	std::unique_ptr<Statement> transformed;
};