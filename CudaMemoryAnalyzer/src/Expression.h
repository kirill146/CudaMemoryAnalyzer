#pragma once
#include "Statement.h"
#include "AnalyzerException.h"

class Expression : public Statement {
public:
	Expression(ExpressionType const& type, clang::SourceLocation const& location);
	virtual ~Expression() = default;
	virtual z3::expr toZ3Expr(State const* state) const = 0;
	virtual std::string getVarName() const { throw AnalyzerException("Expression doesn't have a name"); }
	virtual void rememberMemoryAccesses(State* state, std::vector<MemoryAccess>& memoryAccesses) const { state; memoryAccesses;  }
	ExpressionType getType() const { return type; }
protected:
	ExpressionType type;
};

class IntegerConst : public Expression {
public:
	IntegerConst(uint64_t val, ExpressionType const& type, clang::SourceLocation const& location);
	z3::expr toZ3Expr(State const* state) const override;
public:
	uint64_t val;
};

class RealConst : public Expression {
public:
	RealConst(double val, ExpressionType const& type, clang::SourceLocation const& location);
	z3::expr toZ3Expr(State const* state) const override;
public:
	double val;
};

class BoolConst : public Expression {
public:
	BoolConst(bool val, ExpressionType const& type, clang::SourceLocation const& location);
	z3::expr toZ3Expr(State const* state) const override;
public:
	bool val;
};

class Variable : public Expression {
public:
	Variable(std::string name, ExpressionType const& type, clang::SourceLocation const& location);
	std::string getName() const { return name; }
	std::string getVersionedName(int version) const;
	std::string getVarName() const override { return name; }
	z3::expr toZ3Expr(State const* state) const override;
protected:
	std::string name;
};

class AtomicVariable : public Variable {
public:
	AtomicVariable(std::string name, ExpressionType const& type, uint64_t address,
		clang::SourceLocation const& location);
	//z3::expr toZ3Expr(State const* state) const override;
	z3::expr z3AddressExpr(State const* state) const;
	uint64_t getAddress() const { return address; }
private:
	uint64_t address;
};

uint64_t const UNUSED_ADDR = 512;

class MemberExpression : public Expression {
public:
	MemberExpression(std::unique_ptr<Expression> base, std::string recordName, std::string memberName,
		ExpressionType const& type, clang::SourceLocation const& location);
	z3::expr toZ3Expr(State const* state, z3::expr const& recordVar, std::string const& memberName) const;
	z3::expr toZ3Expr(State const* state) const override;
	Expression const* getBase() const { return base.get(); }
	std::string getRecordName() const { return recordName; }
	std::string getMemberName() const { return memberName; }
private:
	std::unique_ptr<Expression> base;
	std::string recordName;
	std::string memberName;
};

class PodStruct : public Variable {
	PodStruct(std::string name, std::vector<std::unique_ptr<Variable>> fields,
		ExpressionType const& type, clang::SourceLocation const& location);
	z3::expr toZ3Expr(State const* state) const override;
private:
	std::vector<std::unique_ptr<Variable>> fields;
};

class ArraySubscriptExpression : public Expression {
public:
	ArraySubscriptExpression(std::unique_ptr<Expression> base, std::unique_ptr<Expression> index,
		int array_size, ExpressionType const& type, clang::SourceLocation const& location);
	z3::expr toZ3Expr(State const* state) const override;
	void rememberMemoryAccesses(State* state, std::vector<MemoryAccess>& memoryAccesses) const override;
	std::string getVarName() const override;
	int getArraySize() const { return array_size; }
	Expression const* getBase() const { return base.get(); }
	Expression const* getIndex() const { return index.get(); }
private:
	std::unique_ptr<Expression> base;
	std::unique_ptr<Expression> index;
	int array_size;
};

class InitListExpression : public Expression {
public:
	InitListExpression(std::vector<std::unique_ptr<Expression>> initList,
		ExpressionType const& type, clang::SourceLocation const& location);
	z3::expr toZ3Expr(State const* state) const override;
	std::vector<std::unique_ptr<Expression>> const& getInitList() const { return initList; }
private:
	std::vector<std::unique_ptr<Expression>> initList;
};

class CallExpression : public Expression {
public:
	CallExpression(std::string functionName, std::vector<std::unique_ptr<Expression>> arguments,
		ExpressionType const& type, clang::SourceLocation const& location);
	z3::expr toZ3Expr(State const* state) const override;
	std::vector<std::unique_ptr<Expression>> const& getArguments() const { return arguments; }
	std::string getFunctionName() const { return functionName; }
private:
	std::string functionName;
	std::vector<std::unique_ptr<Expression>> arguments;
};

class ImplicitCastExpression : public Expression {
public:
	ImplicitCastExpression(std::unique_ptr<Expression> subExpr, clang::CastKind kind,
		ExpressionType const& type, clang::SourceLocation const& location);
	z3::expr toZ3Expr(State const* state) const override;
	Expression const* getSubExpr() const { return subExpr.get(); }
private:
	std::unique_ptr<Expression> subExpr;
	clang::CastKind kind;
};