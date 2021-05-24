#pragma once
#include "Expression.h"

enum UnaryOperation {
    PRE_INCREMENT,
    PRE_DECREMENT,
    POST_INCREMENT,
    POST_DECREMENT,
    NEGATE,
    NOT, // ~
    ADDR_OF,
    DEREF,
    LNOT, // !
    REAL,
    IMAG,
    UO_UNDEFINED
};

enum BinaryOperation {
    ASSIGN,
    ADD,
    SUB,
    MUL,
    DIV,
    REM,
    SHL,
    SHR,
    EQ,
    NEQ,
    LT,
    LE,
    GT,
    GE,
    LOR,
    OR,
    LAND,
    AND,
    XOR,
    ADD_ASSIGN,
    SUB_ASSIGN,
    MUL_ASSIGN,
    DIV_ASSIGN,
    REM_ASSIGN,
    XOR_ASSIGN,
    BO_UNDEFINED,
    COMMA
};

class UnaryOperator : public Expression {
public:
    UnaryOperator(UnaryOperation op, std::unique_ptr<Expression> arg,
        ExpressionType const& type, clang::SourceLocation const& location);
    z3::expr toZ3Expr(State const* state) const override;
    void modifyState(State* state) const override;
    void rememberMemoryAccesses(State* state, std::vector<MemoryAccess>& memoryAccesses) const override;
    Expression const* getArg() const { return arg.get(); }
    UnaryOperation getOp() const { return op; }
private:
    UnaryOperation op;
    std::unique_ptr<Expression> arg;
};

class BinaryOperator : public Expression {
public:
    BinaryOperator(BinaryOperation op, std::unique_ptr<Expression> lhs,
        std::unique_ptr<Expression> rhs, ExpressionType const& type,
        clang::SourceLocation const& location);
    z3::expr toZ3Expr(State const* state) const override;
    void modifyState(State* state) const override;
    void rememberMemoryAccesses(State* state, std::vector<MemoryAccess>& memoryAccesses) const override;
    Expression const* getLhs() const { return lhs.get(); }
    Expression const* getRhs() const { return rhs.get(); }
private:
    BinaryOperation op;
    std::unique_ptr<Expression> lhs;
    std::unique_ptr<Expression> rhs;
};

class ConditionalOperator : public Expression {
public:
    ConditionalOperator(std::unique_ptr<Expression> cond, std::unique_ptr<Expression> then,
        std::unique_ptr<Expression> _else, ExpressionType const& type,
        clang::SourceLocation const& location);
    z3::expr toZ3Expr(State const* state) const override;
    Expression const* getCond() const { return cond.get(); }
    Expression const* getThen() const { return then.get(); }
    Expression const* getElse() const { return _else.get(); }
private:
    std::unique_ptr<Expression> cond;
    std::unique_ptr<Expression> then;
    std::unique_ptr<Expression> _else;
};