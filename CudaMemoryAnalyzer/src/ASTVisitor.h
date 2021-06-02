#pragma once
#include "AnalyzerContext.h"
#include "State.h"
#include <z3++.h>

class ASTVisitor : public clang::RecursiveASTVisitor<ASTVisitor> {
public:
	ASTVisitor(AnalyzerContext* analyzerContext);
	std::string getFunctionName(clang::Expr const* expr);
	std::unique_ptr<Expression> ProcessCallExpr(clang::CallExpr const* callExpr);
	std::unique_ptr<Expression> ProcessBinaryOperator(clang::BinaryOperator const* binaryOperator);
	std::unique_ptr<Expression> ProcessUnaryOperator(clang::UnaryOperator const* unaryOperator);
	std::unique_ptr<Expression> ProcessMemberExpr(clang::MemberExpr const* memberExpr);
	std::unique_ptr<Expression> ProcessExpr(clang::Expr const* expr);
	std::unique_ptr<Expression> ProcessArraySubscriptExpr(clang::ArraySubscriptExpr const* expr);
	std::unique_ptr<Expression> ProcessFloatingLiteralExpr(clang::FloatingLiteral const* expr);
	std::unique_ptr<Expression> ProcessInitListExpr(clang::InitListExpr const* expr);
	std::unique_ptr<Expression> ProcessImplicitCastExpr(clang::ImplicitCastExpr const* expr);
	std::unique_ptr<Statement> ProcessDecl(clang::Decl* decl);
	std::unique_ptr<Statement> ProcessIfStmt(clang::IfStmt* ifStmt);
	std::unique_ptr<Statement> ProcessWhileStmt(clang::WhileStmt* whileStmt);
	std::unique_ptr<Statement> ProcessForStmt(clang::ForStmt* forStmt);
	std::unique_ptr<Statement> ProcessReturnStmt(clang::ReturnStmt* returnStmt);
	std::unique_ptr<Statement> ProcessStmt(clang::Stmt* stmt);
	std::unique_ptr<Statement> ProcessDeclStmt(clang::DeclStmt* declStmt);
	std::unique_ptr<Statement> ProcessCompoundStmt(clang::CompoundStmt* compoundStmt);
	bool VisitFunctionDecl(clang::FunctionDecl* f);
	bool VisitRecordDecl(clang::RecordDecl const* decl);
	z3::sort getSort(clang::QualType const& type) const;
	z3::sort arrayOfSort(z3::sort const& type) const;
	int getArraySize(clang::QualType const& type) const;
	void AddInitialVarConstraint(clang::QualType type, std::string const& name,
		void const* ptr);
private:
	AnalyzerContext* analyzerContext;
	State* state;
	clang::ASTContext* astContext;
};

class ASTConsumer : public clang::ASTConsumer {
public:
	ASTConsumer(AnalyzerContext* analyzerContext);
	bool HandleTopLevelDecl(clang::DeclGroupRef declarations) override;
private:
	ASTVisitor visitor;
};