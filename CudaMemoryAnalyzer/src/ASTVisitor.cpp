#include "pch.h"
#include "ASTVisitor.h"
#include "Operator.h"
#include "CudaUtils.h"
#include "AnalyzerException.h"

#if 0
#define Log(a) a
#else
#define Log(a)
#endif
std::string cur_fun = "";

z3::sort INT_SORT(State* state) {
	return state->z3_ctx->int_sort();
}

ASTVisitor::ASTVisitor(AnalyzerContext* analyzerContext)
	: analyzerContext(analyzerContext)
	, state(&analyzerContext->ruleContext.state)
{}

std::string ASTVisitor::getFunctionName(clang::Expr const* expr) {
	if (clang::isa<clang::ImplicitCastExpr>(expr)) {
		return getFunctionName(clang::cast<clang::ImplicitCastExpr>(expr)->getSubExpr());
	}
	if (clang::isa<clang::DeclRefExpr>(expr)) {
		return clang::cast<clang::DeclRefExpr>(expr)->getNameInfo().getAsString();
	}
	if (clang::isa<clang::MemberExpr>(expr)) {
		return clang::cast<clang::MemberExpr>(expr)->getMemberNameInfo().getAsString();
	}
	std::string className = expr->getStmtClassName();
	throw AnalyzerException(("unexpected expr in getFunctionName(): " + className).c_str());
}

std::unique_ptr<Expression> ASTVisitor::ProcessCallExpr(clang::CallExpr const* callExpr) {
	clang::QualType qualType = callExpr->getType();
	ExpressionType type(qualType, getSort(qualType), getArraySize(qualType));
	if (clang::isa<clang::CXXOperatorCallExpr>(callExpr) &&
		clang::cast<clang::CXXOperatorCallExpr>(callExpr)->isAssignmentOp())
	{
		auto lhs = ProcessExpr(callExpr->getArg(0));
		Log(std::cout << " = ";);
		auto rhs = ProcessExpr(callExpr->getArg(1));
		return std::make_unique<BinaryOperator>(ASSIGN, std::move(lhs), std::move(rhs), type,
			callExpr->getExprLoc());
	}
	std::string functionName = getFunctionName(callExpr->getCallee());
	Log(std::cout << functionName << "(");
	int numArgs = callExpr->getNumArgs();
	std::vector<std::unique_ptr<Expression>> arguments;
	for (int i = 0; i < numArgs; i++) {
		arguments.push_back(ProcessExpr(callExpr->getArg(i)));
		if (i != numArgs - 1) {
			Log(std::cout << ", ");
		}
	}
	Log(std::cout << ")");
	return std::make_unique<CallExpression>(functionName, std::move(arguments), type,
		callExpr->getExprLoc());
}

std::unique_ptr<Expression> ASTVisitor::ProcessBinaryOperator(clang::BinaryOperator const* binaryOperator) {
	auto lhs = ProcessExpr(binaryOperator->getLHS());
	auto opCode = binaryOperator->getOpcode();
	Log(std::cout << ' ' << binaryOperator->getOpcodeStr().str() << ' ');
	auto rhs = ProcessExpr(binaryOperator->getRHS());
	clang::QualType qualType = binaryOperator->getType();
	ExpressionType type(qualType, getSort(qualType), getArraySize(qualType));
	static std::unordered_map<clang::BinaryOperator::Opcode, BinaryOperation> const opMap = {
		{ clang::BO_Assign, ASSIGN },
		{ clang::BO_Add, ADD },
		{ clang::BO_Sub, SUB },
		{ clang::BO_Mul, MUL },
		{ clang::BO_Div, DIV },
		{ clang::BO_Rem, REM },
		{ clang::BO_Shl, SHL },
		{ clang::BO_Shr, SHR },
		{ clang::BO_EQ, EQ },
		{ clang::BO_NE, NEQ },
		{ clang::BO_LT, LT },
		{ clang::BO_LE, LE },
		{ clang::BO_GT, GT },
		{ clang::BO_GE, GE },
		{ clang::BO_LOr, LOR },
		{ clang::BO_Or, OR },
		{ clang::BO_LAnd, LAND },
		{ clang::BO_And, AND },
		{ clang::BO_Xor, XOR },
		{ clang::BO_AddAssign, ADD_ASSIGN },
		{ clang::BO_SubAssign, SUB_ASSIGN },
		{ clang::BO_MulAssign, MUL_ASSIGN },
		{ clang::BO_DivAssign, DIV_ASSIGN },
		{ clang::BO_RemAssign, REM_ASSIGN },
		{ clang::BO_XorAssign, XOR_ASSIGN },
		{ clang::BO_Comma, COMMA }
	};
	if (opMap.find(opCode) != opMap.end()) {
		return std::make_unique<BinaryOperator>(opMap.at(opCode), std::move(lhs), std::move(rhs),
			type, binaryOperator->getOperatorLoc());
	}
	PEDANTIC_THROW("Undefined binary operator at ProcessBinaryOperator()");
	return nullptr;
}

std::unique_ptr<Expression> ASTVisitor::ProcessUnaryOperator(clang::UnaryOperator const* unaryOperator) {
	auto opCode = unaryOperator->getOpcode();
	Log(std::cout << unaryOperator->getOpcodeStr(opCode).str());
	auto arg = ProcessExpr(unaryOperator->getSubExpr());
	if (opCode == clang::UO_Plus) {
		return arg;
	}
	clang::QualType qualType = unaryOperator->getType();
	ExpressionType type(qualType, getSort(qualType), getArraySize(qualType));
	static std::unordered_map<clang::UnaryOperator::Opcode, UnaryOperation> const opMap = {
		{ clang::UO_PostInc, POST_INCREMENT },
		{ clang::UO_PostDec, POST_DECREMENT },
		{ clang::UO_PreInc, PRE_INCREMENT },
		{ clang::UO_PreDec, PRE_DECREMENT },
		{ clang::UO_Minus, NEGATE },
		{ clang::UO_LNot, LNOT },
		{ clang::UO_AddrOf, ADDR_OF },
		{ clang::UO_Deref, DEREF },
		{ clang::UO_Not, NOT },
		{ clang::UO_Real, REAL },
		{ clang::UO_Imag, IMAG }
	};
	
	if (opMap.find(opCode) != opMap.end()) {
		return std::make_unique<UnaryOperator>(opMap.at(opCode), std::move(arg), type,
			unaryOperator->getOperatorLoc());
	}
	PEDANTIC_THROW("Undefined unary operator at ProcessUnaryOperator()");
	return nullptr;
}

std::unique_ptr<Expression> ASTVisitor::ProcessMemberExpr(clang::MemberExpr const* memberExpr) {
	clang::QualType qualType = memberExpr->getType();
	ExpressionType type(qualType, getSort(qualType), getArraySize(qualType));
	auto base = ProcessExpr(memberExpr->getBase());
	if (base == nullptr) {
		return nullptr;
	}
	clang::QualType baseType = base->getType().type;
	//std::cout << "actual type: " << baseType.getAsString() << std::endl;
	if (baseType->isPointerType()) {
		Log(std::cout << "->");
	} else if (baseType->isRecordType()) {
		Log(std::cout << '.');
	} else {
		throw AnalyzerException("Expected Record or Pointer type");
	}
	
	auto decl = memberExpr->getMemberDecl();
	if (!clang::isa<clang::FieldDecl>(decl)) {
		throw AnalyzerException("Expected FieldDecl");
	}
	auto fieldDecl = clang::cast<clang::FieldDecl>(decl);
	std::string fieldName = fieldDecl->getName().str();
	Log(std::cout << fieldName);
	if (baseType->isPointerType()) {
		return nullptr;
	}
	//std::cout << "just: " << base->getType().type.getAsString() << std::endl;
	//std::cout << "canonical: " << base->getType().type.getCanonicalType().getAsString() << std::endl;
	//std::cout << "base type class name: " << base->getType().type->isRecordType() << ' ' << clang::isa<clang::RecordType>(base->getType().type) << std::endl;
	auto recordDecl = clang::cast<clang::RecordType>(baseType.getCanonicalType())->getDecl();
	std::string recordName = recordDecl->getName().str();
	//ExpressionType dreType(baseType, getSort(baseType),	getArraySize(baseType));
	return std::make_unique<MemberExpression>(
		std::move(base),
		recordName,
		fieldName,
		type,
		memberExpr->getBeginLoc()
	);
	/*clang::QualType pt = baseType->getPointeeType();
	int array_size = getArraySize(pt);
	ExpressionType pointeeType(pt, getSort(pt), array_size);*/
}

std::unique_ptr<Expression> ASTVisitor::ProcessExpr(clang::Expr const* expr) {
	clang::QualType qualType = expr->getType();
	ExpressionType type(qualType, getSort(qualType), getArraySize(qualType));
	if (clang::isa<clang::CallExpr>(expr)) {
		return ProcessCallExpr(clang::cast<clang::CallExpr>(expr));
	}
	if (clang::isa<clang::SubstNonTypeTemplateParmExpr>(expr)) {
		return ProcessExpr(clang::cast<clang::SubstNonTypeTemplateParmExpr>(expr)->getReplacement());
	}
	if (clang::isa<clang::CXXConstructExpr>(expr)) {
		Log(auto constructExpr = clang::cast<clang::CXXConstructExpr>(expr);)
		//Log(std::cout << constructExpr->getConstructor()->getID() << std::endl);
		Log(std::cout << constructExpr->getConstructor()->getDeclName().getAsString());// getAsFunction()->getName().str() << std::endl;
		Log(std::cout << "[CXXConstructExpr]");
		return nullptr;
	}
	if (clang::isa<clang::MemberExpr>(expr)) {
		return ProcessMemberExpr(clang::cast<clang::MemberExpr>(expr));
	}
	if (clang::isa<clang::UnaryExprOrTypeTraitExpr>(expr)) {
		auto sizeofExpr = clang::cast<clang::UnaryExprOrTypeTraitExpr>(expr);
		if (!sizeofExpr->isEvaluatable(*astContext)) {
			throw AnalyzerException("!isEvaluatable()");
		}
		clang::Expr::EvalResult result;
		if (!sizeofExpr->EvaluateAsInt(result, *astContext)) {
			throw AnalyzerException("Cannot evaluate sizeof()");
		}
		return std::make_unique<IntegerConst>(
			result.Val.getInt().getLimitedValue(),
			ExpressionType(qualType, INT_SORT(state), -1), expr->getExprLoc());
	}
	if (clang::isa<clang::BinaryOperator>(expr)) {
		return ProcessBinaryOperator(clang::cast<clang::BinaryOperator>(expr));
	}
	if (clang::isa<clang::ImplicitCastExpr>(expr)) {
		return ProcessExpr(clang::cast<clang::ImplicitCastExpr>(expr)->getSubExpr());
		//return ProcessImplicitCastExpr(clang::cast<clang::ImplicitCastExpr>(expr));
	}
	if (clang::isa<clang::ExplicitCastExpr>(expr)) {
		Log(std::cout << "(Cast_type)");
		return ProcessExpr(clang::cast<clang::ExplicitCastExpr>(expr)->getSubExpr());
	}
	if (clang::isa<clang::ArraySubscriptExpr>(expr)) {
		return ProcessArraySubscriptExpr(clang::cast<clang::ArraySubscriptExpr>(expr));
	}
	if (clang::isa<clang::IntegerLiteral>(expr)) {
		uint64_t val = clang::cast<clang::IntegerLiteral>(expr)->getValue().getLimitedValue();
		Log(std::cout << val);
		return std::make_unique<IntegerConst>(val, type, expr->getExprLoc());
	}
	if (clang::isa<clang::CharacterLiteral>(expr)) {
		char val = (char)clang::cast<clang::CharacterLiteral>(expr)->getValue();
		Log(std::cout << "'" << val << "'");
		return std::make_unique<IntegerConst>(val, type, expr->getExprLoc());
	}
	if (clang::isa<clang::FloatingLiteral>(expr)) {
		return ProcessFloatingLiteralExpr(clang::cast<clang::FloatingLiteral>(expr));
	}
	if (clang::isa<clang::CXXBoolLiteralExpr>(expr)) {
		bool val = clang::cast<clang::CXXBoolLiteralExpr>(expr)->getValue();
		if (val) {
			Log(std::cout << "true");
		} else {
			Log(std::cout << "false");
		}
		return std::make_unique<BoolConst>(val, type, expr->getExprLoc());
	}
	if (clang::isa<clang::UnaryOperator>(expr)) {
		return ProcessUnaryOperator(clang::cast<clang::UnaryOperator>(expr));
	}
	if (clang::isa<clang::ParenExpr>(expr)) {
		return ProcessExpr(clang::cast<clang::ParenExpr>(expr)->getSubExpr());
	}
	if (clang::isa<clang::ConditionalOperator>(expr)) {
		auto conditionalOperator = clang::cast<clang::ConditionalOperator>(expr);
		std::unique_ptr<Expression> cond = ProcessExpr(conditionalOperator->getCond());
		Log(std::cout << " ? ");
		std::unique_ptr<Expression> _true = ProcessExpr(conditionalOperator->getTrueExpr());
		Log(std::cout << " : ");
		std::unique_ptr<Expression> _false = ProcessExpr(conditionalOperator->getFalseExpr());
		return std::make_unique<ConditionalOperator>(
			std::move(cond), std::move(_true), std::move(_false), type, expr->getExprLoc());
	}
	if (clang::isa<clang::DeclRefExpr>(expr)) {
		auto declRef = clang::cast<clang::DeclRefExpr>(expr);
		std::string name = declRef->getNameInfo().getAsString();
		Log(std::cout << name);
		//std::cout << "[type: " << name << "]";
		auto savedType = state->getVariableType(name);
		if (savedType) {
			type = *savedType;
		}
		return std::make_unique<AtomicVariable>(name, type, UNUSED_ADDR, expr->getExprLoc());
	}
	if (clang::isa<clang::InitListExpr>(expr)) {
		return ProcessInitListExpr(clang::cast<clang::InitListExpr>(expr));
	}
	if (clang::isa<clang::ExprWithCleanups>(expr)) {
		Log(std::cout << "ExprWithCleanups\n");
		return nullptr;
	}
	if (clang::isa<clang::PseudoObjectExpr>(expr)) {
		auto const& poe = clang::cast<clang::PseudoObjectExpr>(expr);
			if (clang::isa<clang::CallExpr>(poe->getResultExpr())) {
				auto ce = clang::cast<clang::CallExpr>(poe->getResultExpr());
				if (clang::isa<clang::ImplicitCastExpr>(ce->getCallee())) {
					auto ice = clang::cast<clang::ImplicitCastExpr>(ce->getCallee());
					if (clang::isa<clang::MemberExpr>(ice->getSubExpr())) {
						auto me = clang::cast<clang::MemberExpr>(ice->getSubExpr());
						std::string methodName = me->getMemberNameInfo().getAsString();
						if (clang::isa<clang::OpaqueValueExpr>(me->getBase())) {
							auto ove = clang::cast<clang::OpaqueValueExpr>(me->getBase());
							if (clang::isa<clang::DeclRefExpr>(ove->getSourceExpr())) {
								auto dre = clang::cast<clang::DeclRefExpr>(ove->getSourceExpr());
								std::string varName = dre->getNameInfo().getAsString();
								if ((varName == "threadIdx" || varName == "blockIdx" || varName == "gridDim" || varName == "blockDim") &&
									(methodName == "__fetch_builtin_x" || methodName == "__fetch_builtin_y" || methodName == "__fetch_builtin_z"))
								{
									std::string name = varName + "." + methodName[methodName.length() - 1];
									return std::make_unique<AtomicVariable>(name, type, UNUSED_ADDR, expr->getExprLoc());
								}
							}
						}
						return nullptr;
					}
				}
			}
		return ProcessExpr(clang::cast<clang::PseudoObjectExpr>(expr)->getResultExpr());
	}
	if (clang::isa<clang::OpaqueValueExpr>(expr)) {
		Log(std::cout << "OpaqueValueExpr");
		return ProcessExpr(clang::cast<clang::OpaqueValueExpr>(expr)->getSourceExpr());
	}
	Log(std::cout << expr->getStmtClassName() << std::endl);
	throw AnalyzerException("Unknown expression " + std::string(expr->getStmtClassName()));
}

std::string getArrayName(clang::Expr const* expr) {
	if (clang::isa<clang::DeclRefExpr>(expr)) {
		return clang::cast<clang::DeclRefExpr>(expr)->getNameInfo().getAsString();
	} else if (clang::isa<clang::MemberExpr>(expr)) {
		return "[Unparsed MemberExpr]";
	} else if (clang::cast<clang::ArraySubscriptExpr>(expr)) {
		return "[Unparsed ArraySubscriptExpr]";
	}
	throw AnalyzerException("Unknown base type of ArraySubscriptExpression");
}

std::unique_ptr<Expression> ASTVisitor::ProcessArraySubscriptExpr(clang::ArraySubscriptExpr const* expr) {
	clang::Expr const* base = expr->getBase();
	if (clang::isa<clang::ImplicitCastExpr>(base)) {
		base = clang::cast<clang::ImplicitCastExpr>(base)->getSubExpr();
	}
	std::unique_ptr<Expression> base_res = ProcessExpr(base);
	Log(std::cout << "[");
	std::unique_ptr<Expression> index_res = ProcessExpr(expr->getIdx());
	Log(std::cout << "]");
	clang::QualType qualType = expr->getType();
	ExpressionType type(qualType, getSort(qualType), getArraySize(qualType));
	int array_size = getArraySize(base->getType());
	std::string array_name = getArrayName(base);
	//bool dynamicSharedArray = false;
	clang::VarDecl const* dynamicSharedArray = nullptr;
	if (clang::isa<clang::DeclRefExpr>(base)) {
		clang::DeclRefExpr const* dre = clang::cast<clang::DeclRefExpr>(base);
		if (clang::isa<clang::VarDecl>(dre->getDecl())) {
			clang::VarDecl const* varDecl = clang::cast<clang::VarDecl>(dre->getDecl());
			if (varDecl->hasExternalStorage() && varDecl->hasAttr<clang::CUDASharedAttr>()
				&& varDecl->getType()->isArrayType())
			{
				dynamicSharedArray = varDecl;
			}
		}
	}
	if (array_size == -1 && analyzerContext->kernelContext.argSizes.count(array_name)) {
		array_size = (int)analyzerContext->kernelContext.argSizes[array_name];
	}
	if (array_size == -1 && dynamicSharedArray != nullptr) {
		int elemSize = (int)astContext->getTypeSize(dynamicSharedArray->getType()->getArrayElementTypeNoTypeQual()) / 8;
		array_size = (int)analyzerContext->kernelContext.dynamicSharedMemSize / elemSize;
		//Log(std::cout << "dsa_size: " << array_size << std::endl << "elem_sz: " << elemSize << std::endl);
	}
	return std::make_unique<ArraySubscriptExpression>(std::move(base_res), std::move(index_res), array_size,
		type, expr->getIdx()->getExprLoc());
}

std::unique_ptr<Expression> ASTVisitor::ProcessFloatingLiteralExpr(clang::FloatingLiteral const* expr) {
	clang::QualType qualType = expr->getType();
	ExpressionType type(qualType, getSort(qualType), getArraySize(qualType));
	llvm::APFloatBase::Semantics semantics = expr->getRawSemantics();
	double val;
	if (semantics == llvm::APFloatBase::S_IEEEdouble) {
		val = expr->getValue().convertToDouble();
		Log(std::cout << val);
	} else if (semantics == llvm::APFloatBase::S_IEEEsingle) {
		val = expr->getValue().convertToFloat();
		Log(std::cout << val << 'f');
	} else {
		throw AnalyzerException("Unexpected type of floating point literal");
	}
	return std::make_unique<RealConst>(val, type, expr->getEndLoc());
}

std::unique_ptr<Expression> ASTVisitor::ProcessInitListExpr(clang::InitListExpr const* expr) {
	clang::QualType qualType = expr->getType();
	ExpressionType type(qualType, getSort(qualType), getArraySize(qualType));
	std::vector<std::unique_ptr<Expression>> expressions;
	Log(std::cout << "{ ");
	for (auto& e : expr->inits()) {
		std::unique_ptr<Expression> expression = ProcessExpr(e);
		Log(std::cout << ", ");
		expressions.push_back(std::move(expression));
	}
	Log(std::cout << "}");
	return std::make_unique<InitListExpression>(std::move(expressions), type, expr->getExprLoc());
}

std::unique_ptr<Expression> ASTVisitor::ProcessImplicitCastExpr(clang::ImplicitCastExpr const* expr) {
	clang::QualType qualType = expr->getType();
	ExpressionType type(qualType, getSort(qualType), getArraySize(qualType));
	Log(std::cout << "(" << expr->getCastKindName() << ")");
	return std::make_unique<ImplicitCastExpression>(ProcessExpr(expr->getSubExpr()), expr->getCastKind(), type, expr->getExprLoc());
}

std::unique_ptr<Statement> ASTVisitor::ProcessDecl(clang::Decl* decl) {
	if (clang::isa<clang::CXXRecordDecl>(decl)) {
		Log(std::cout << clang::cast<clang::CXXRecordDecl>(decl)->getName().str().c_str());
		return nullptr;
	}
	if (clang::isa<clang::StaticAssertDecl>(decl)) {
		Log(std::cout << "[static_assert: ignored]");
		return nullptr;
	}
	if (clang::isa<clang::TypedefDecl>(decl)) {
		Log(std::cout << "[Unparsed typedef]");
		return nullptr;
	}
	if (clang::isa<clang::VarDecl>(decl)) {
		auto varDecl = clang::cast<clang::VarDecl>(decl);
		if (varDecl->hasExternalStorage()) {
			Log(std::cout << "extern ");
		}
		if (varDecl->hasAttr<clang::CUDASharedAttr>()) {
			Log(std::cout << "__shared__ ");
		}
		Log(std::cout << varDecl->getType().getAsString() << ' ');
		std::string name = varDecl->getName().str();
		Log(std::cout << name);
		clang::Expr const* initializer = varDecl->getAnyInitializer();
		if (initializer && clang::isa<clang::CXXConstructExpr>(initializer)) {
			auto cxxce = clang::cast<clang::CXXConstructExpr>(initializer);
			if (cxxce->getConstructor()->isCopyConstructor()) {
				initializer = cxxce->getArg(0);
			} else {
				initializer = nullptr;
			}
		}
		if (initializer) {
			Log(std::cout << " = ");
		}
		clang::QualType qualType = varDecl->getType();
		ExpressionType type(qualType, getSort(qualType), getArraySize(qualType));
		clang::SourceLocation location = decl->getLocation();
		uint64_t size = astContext->getTypeSize(qualType) / 8;
		
		return std::make_unique<DeclStatement>(
			std::make_unique<AtomicVariable>(
				name, type, analyzerContext->ruleContext.allocateMemory(size), location),
			initializer ? ProcessExpr(initializer) : nullptr, location);
	}
	PEDANTIC_THROW("Unexpected declaration");
	return nullptr;
}

std::unique_ptr<Statement> ASTVisitor::ProcessIfStmt(clang::IfStmt* ifStmt) {
	Log(std::cout << "if (");
	std::unique_ptr<Expression> cond = ProcessExpr(ifStmt->getCond());
	Log(std::cout << ") {" << std::endl);
	std::unique_ptr<Statement> then = ProcessStmt(ifStmt->getThen());
	Log(std::cout << "}");
	std::unique_ptr<Statement> _else = nullptr;
	if (ifStmt->hasElseStorage()) {
		Log(std::cout << " else {" << std::endl);
		_else = ProcessStmt(ifStmt->getElse());
		Log(std::cout << "}");
	}
	Log(std::cout << std::endl);
	return std::make_unique<IfStatement>(std::move(cond), std::move(then), std::move(_else),
		ifStmt->getIfLoc());
}

std::unique_ptr<Statement> ASTVisitor::ProcessWhileStmt(clang::WhileStmt* whileStmt) {
	Log(std::cout << "while(");
	auto cond = ProcessExpr(whileStmt->getCond());
	Log(std::cout << ") {\n");
	auto body = ProcessStmt(whileStmt->getBody());
	Log(std::cout << "}");
	return std::make_unique<WhileStatement>(std::move(cond), std::move(body),
		whileStmt->getWhileLoc());
}

std::unique_ptr<Statement> ASTVisitor::ProcessForStmt(clang::ForStmt* forStmt) {
	Log(std::cout << "for (");
	auto init = ProcessStmt(forStmt->getInit());
	Log(std::cout << "; ");
	auto cond = ProcessExpr(forStmt->getCond());
	Log(std::cout << "; ");
	auto inc = ProcessExpr(forStmt->getInc());
	Log(std::cout << ") {\n");
	auto body = ProcessStmt(forStmt->getBody());
	Log(std::cout << "}");
	return std::make_unique<ForStatement>(std::move(init), std::move(cond), std::move(inc),
		std::move(body), forStmt->getForLoc());
}

std::unique_ptr<Statement> ASTVisitor::ProcessReturnStmt(clang::ReturnStmt* returnStmt) {
	Log(std::cout << "return ");
	clang::Expr* retVal = returnStmt->getRetValue();
	return std::make_unique<ReturnStatement>(
		retVal ? ProcessExpr(returnStmt->getRetValue()) : nullptr, returnStmt->getReturnLoc());
}

std::unique_ptr<Statement> ASTVisitor::ProcessStmt(clang::Stmt* stmt) {
	if (stmt == nullptr || clang::isa<clang::NullStmt>(stmt)) {
		return nullptr;
	}
	if (clang::isa<clang::AttributedStmt>(stmt)) {
		auto as = clang::cast<clang::AttributedStmt>(stmt);
		return ProcessStmt(as->getSubStmt());
	}
	if (clang::isa<clang::IfStmt>(stmt)) {
		return ProcessIfStmt(clang::cast<clang::IfStmt>(stmt));
	}
	if (clang::isa<clang::WhileStmt>(stmt)) {
		return ProcessWhileStmt(clang::cast<clang::WhileStmt>(stmt));
	}
	if (clang::isa<clang::ForStmt>(stmt)) {
		return ProcessForStmt(clang::cast<clang::ForStmt>(stmt));
	}
	if (clang::isa<clang::CompoundStmt>(stmt)) {
		return ProcessCompoundStmt(clang::cast<clang::CompoundStmt>(stmt));
	}
	if (clang::isa<clang::DeclStmt>(stmt)) {
		return ProcessDeclStmt(clang::cast<clang::DeclStmt>(stmt));
	}
	if (clang::isa<clang::ReturnStmt>(stmt)) {
		return ProcessReturnStmt(clang::cast<clang::ReturnStmt>(stmt));
	}
	if (clang::isa<clang::GCCAsmStmt>(stmt)) {
		Log(std::cout << "[unparsed ASM]" << std::endl);
		return nullptr;
	}
	if (clang::isa<clang::Expr>(stmt)) {
		return ProcessExpr(clang::cast<clang::Expr>(stmt));
	}

	PEDANTIC_THROW("Unknown statement " + std::string(stmt->getStmtClassName()));
	return nullptr;
}

std::unique_ptr<Statement> ASTVisitor::ProcessDeclStmt(clang::DeclStmt* declStmt) {
	std::vector<std::unique_ptr<Statement>> statements;
	for (auto decl = declStmt->decl_begin(); decl != declStmt->decl_end(); decl++) {
		auto statement = ProcessDecl(*decl);
		if (statement != nullptr) {
			statements.push_back(std::move(statement));
		}
	}
	return std::make_unique<CompoundStatement>(std::move(statements), declStmt->getBeginLoc());
}

std::unique_ptr<Statement> ASTVisitor::ProcessCompoundStmt(clang::CompoundStmt* compoundStmt) {
	std::vector<std::unique_ptr<Statement>> statements;
	for (auto stmt = compoundStmt->body_begin(); stmt != compoundStmt->body_end(); stmt++) {
		auto statement = ProcessStmt(*stmt);
		if (statement != nullptr) {
			statements.push_back(std::move(statement));
		}
		Log(std::cout << ";\n");
	}
	return std::make_unique<CompoundStatement>(std::move(statements), compoundStmt->getBeginLoc());
}

void ASTVisitor::AddInitialVarConstraint(clang::QualType type, std::string const& name,
	void const* ptr)
{
	z3::expr var = state->z3_ctx->constant((name + "!0").c_str(), getSort(type));
	state->variableVersions[name] = state->acquireNextVersion(name);
	int typeSize = (int)astContext->getTypeSize(type);
	if (type->isIntegerType()) {
		switch (typeSize) {
		case 8: {
			int8_t value = *(int8_t*)ptr;
			state->predicates.push_back(z3::expr(var == value));
			break;
		}
		case 16: {
			int16_t value = *(int16_t*)ptr;
			state->predicates.push_back(z3::expr(var == value));
			break;
		}
		case 32: {
			int32_t value = *(int32_t*)ptr;
			state->predicates.push_back(z3::expr(var == value));
			break;
		}
		case 64: {
			z3::expr value = state->z3_ctx->int_val(*(int64_t*)ptr);
			state->predicates.push_back(z3::expr(var == value));
			break;
		}
		default:
			throw AnalyzerException("Unknown integer type argument");
		}
	} else if (type->isFloatingType()) {
		switch (typeSize) {
		case 32: {
			z3::expr value = state->z3_ctx->real_val(std::to_string(*(float*)ptr).c_str());
			state->predicates.push_back(z3::expr(var == value));
			break;
		}
		case 64: {
			z3::expr value = state->z3_ctx->real_val(std::to_string(*(double*)ptr).c_str());
			state->predicates.push_back(z3::expr(var == value));
			break;
		}
		default:
			throw AnalyzerException("Unknown floating type argument");
		}
	}
}

bool ASTVisitor::VisitFunctionDecl(clang::FunctionDecl* f) {
	astContext = &f->getASTContext();
	std::string functionName = f->getNameInfo().getName().getAsString();
	if (f->hasAttr<clang::CUDAGlobalAttr>() &&
		functionName != analyzerContext->kernelContext.kernelName)
	{
		Log(std::cout << "[Unparsed global function]" << std::endl);
		return true;
	}
	if (!f->hasAttr<clang::CUDADeviceAttr>() && !f->hasAttr<clang::CUDAGlobalAttr>()) {
		Log(std::cout << "[Unparsed host function " << functionName << "]" << std::endl);
		return true;
	}
	if (!f->isThisDeclarationADefinition()) {
		Log(std::cout << "[Unparsed function declaration]" << std::endl);
		return true;
	}
	//if (clang::isa<clang::FunctionTemplateDecl>(f)) {
	if (f->isTemplated()) {
		return true;
		//auto td = ftd->getTemplatedDecl();
		//auto tpl = ftd->getTemplateParameters();
		#if 0
		auto tpl = f->getDescribedTemplateParams();
		
		if (tpl == nullptr) {
			Log(std::cout << "[Unparsed templated function]" << std::endl);
			return true;
		}
		bool allTemplateParamsAreNonTypes = true;
		for (int i = 0; i < (int)tpl->size(); i++) {
			if (!clang::isa<clang::NonTypeTemplateParmDecl>(tpl->getParam(i))) {
				allTemplateParamsAreNonTypes = false;
				break;
			}
		}
		if (!allTemplateParamsAreNonTypes) {
			Log(std::cout << "[Unparsed templated function]" << std::endl);
			return true;
		}
		Log(std::cout << "template <");
		for (int i = 0; i < (int)tpl->size(); i++) {
			auto nttpd = clang::cast<clang::NonTypeTemplateParmDecl>(tpl->getParam(i));
			std::string name = nttpd->getName().str();
			clang::QualType type = nttpd->getType();
			if (i != 0) {
				Log(std::cout << ", ";)
			}
			Log(std::cout << type.getAsString() << ' ' << name);
			AddInitialVarConstraint(type, name, &analyzerContext->kernelContext.templateArgValues[i]);
			/*if (type->isIntegerType()) {
				int typeSize = (int)astContext->getTypeSize(type);
				if (typeSize == 32) {
					std::cout << "int\n";
				} else if (typeSize == 64) {
					std::cout << "size_t\n";
				}
			} else {
				std::cout << "smth else\n";
			}*/
		}
		Log(std::cout << ">\n");
		#endif
	}
	//bool foundGlobalInstantiation = false;
	if (f->isTemplateInstantiation()) {
		if (!f->hasAttr<clang::CUDAGlobalAttr>()) {
			Log(std::cout << "[Unparesed templated device function]\n");
			return true;
		}
		auto tsa = f->getTemplateSpecializationArgs();
		if (tsa == nullptr) {
			std::cout << "no tsa\n";
			return true;
		}

		bool allArgsAreIntegral = true;
		for (int i = 0; i < (int)tsa->size(); i++) {
			if (tsa->get(i).getKind() != clang::TemplateArgument::ArgKind::Integral) {
				allArgsAreIntegral = false;
				break;
			}
		}
		if (!allArgsAreIntegral) {
			return true;
		}

		std::vector<uint64_t> templateArgs;
		for (int i = 0; i < (int)tsa->size(); i++) {
			templateArgs.push_back(tsa->get(i).getAsIntegral().getLimitedValue());
		}
		if (templateArgs == analyzerContext->kernelContext.templateArgValues) {
			Log(std::cout << "FOUND correct global instantiation\n");
			//foundGlobalInstantiation = true;
		} else {
			Log(std::cout << "[Unparsed templated global function instantiation]" << std::endl);
			return true;
		}
	}
	if (f->isCXXClassMember()) {
		Log(std::cout << "[Unparsed class member]" << std::endl);
		return true;
	}
	/*std::string str_loc;
	str_loc = f->getSourceRange().getBegin().printToString(f->getASTContext().getSourceManager());
	Log(std::cout << "str_loc: " << str_loc << std::endl);
	*/
	if (f->hasAttr<clang::CUDAGlobalAttr>()) {
		Log(std::cout << "__global__ ");
	}
	Log(std::cout << f->getReturnType().getAsString() << ' ');
	Log(std::cout << functionName << '(');
	
	int paramsCnt = f->getNumParams();
	std::vector<std::unique_ptr<AtomicVariable>> arguments;
	for (int i = 0; i < paramsCnt; i++) {
		clang::ParmVarDecl* paramDecl = f->getParamDecl(i);
		clang::QualType type = paramDecl->getType();
		std::string name = paramDecl->getNameAsString();
		Log(std::cout << type.getAsString() << ' ' << name);
		if (i != paramsCnt - 1) {
			Log(std::cout << ", ");
		}
		if (f->hasAttr<clang::CUDAGlobalAttr>()) {
			AddInitialVarConstraint(type, name, analyzerContext->kernelContext.scalarArgValues[i]);
			int array_size = getArraySize(type);
			auto sort = getSort(type);
			if (type->isPointerType()) {
				auto elemType = type->getPointeeType();
				size_t& size = analyzerContext->kernelContext.argSizes[name];
				size_t sz = BufSizeByAddress(analyzerContext->kernelContext.scalarArgValues[i]);
				size = sz * 8 / (int)astContext->getTypeSize(elemType);
				array_size = (int)size;
				sort = arrayOfSort(getSort(elemType));
			}
			auto exprType = ExpressionType(type, sort, array_size);
			AtomicVariable variable(name, exprType,
				(uint64_t)analyzerContext->kernelContext.scalarArgValues[i], paramDecl->getLocation());
			state->variableTypes.insert({ name, exprType });
			state->localVariablesPredicates.emplace_back(
				variable.z3AddressExpr(state) == state->z3_ctx->int_val(variable.getAddress())
			);
		} else {
			ExpressionType expressionType(type, getSort(type), getArraySize(type));
			arguments.push_back(std::make_unique<AtomicVariable>(
				name, expressionType, UNUSED_ADDR, paramDecl->getLocation()));
		}
	}
	Log(std::cout << ")");
	if (f->hasBody()) {
		Log(std::cout << " {" << std::endl);
		clang::Stmt* body = f->getBody();
		if (f->hasAttr<clang::CUDAGlobalAttr>()) {
			cur_fun = f->getName().str();
			analyzerContext->kernelBody = ProcessStmt(body);
		} else {
			analyzerContext->ruleContext.functions.emplace(
				functionName, Function(functionName, std::move(arguments), ProcessStmt(body)));
		}
		Log(std::cout << "}" << std::endl);
	} else {
		Log(std::cout << ";" << std::endl);
	}
	return true;
}

bool ASTVisitor::VisitRecordDecl(clang::RecordDecl const* decl) {
	if (!decl->isCompleteDefinition()) {
		return true;
	}
	if (clang::isa<clang::CXXRecordDecl>(decl)) {
		auto cxxDecl = clang::cast<clang::CXXRecordDecl>(decl);
		if (!cxxDecl->isThisDeclarationADefinition() ||
			!cxxDecl->isStruct() ||
			!cxxDecl->isTrivial() ||
			!cxxDecl->isPOD() ||
			cxxDecl->method_begin() != cxxDecl->method_end()) {
			Log(std::cout << "[Unparsed struct]\n");
			return true;
		}
		/*
		for (auto const& method : cxxDecl->methods()) {
			std::cout << "\t method: " << method->getNameAsString() << std::endl;
		}*/

		Log(std::cout << "rrstruct " << cxxDecl->getName().str() << " {\n");
		for (auto it = cxxDecl->field_begin(); it != cxxDecl->field_end(); ++it) {
			auto field = *it;
			clang::QualType qualType = field->getType();
			//Variable
			Log(std::cout << '\t' << qualType.getAsString() << ' ' << field->getName().str() << ";\n");
		}
		Log(std::cout << "}\n");
	}
	return true;
}

z3::sort ASTVisitor::arrayOfSort(z3::sort const& sort) const {
	return state->z3_ctx->array_sort(INT_SORT(state), sort);
}

z3::sort ASTVisitor::getSort(clang::QualType const& type) const {
	if (type->isBooleanType()) {
		return state->z3_ctx->bool_sort();
	}
	if (type->isIntegerType() || type->isPointerType()) {
		return state->z3_ctx->int_sort();
	}
	if (type->isFloatingType()) {
		return state->z3_ctx->real_sort();
	}
	//if (type->isConstantArrayType()) {
	if (clang::isa<clang::ConstantArrayType>(type)) {
		return arrayOfSort(getSort(clang::cast<clang::ConstantArrayType>(type)->getElementType()));
	}
	if (type->isExtVectorType()) {
		auto arrayType = clang::cast<clang::ExtVectorType>(type.getDesugaredType(*astContext));
		return arrayOfSort(getSort(arrayType->getElementType()));
	}
	if (type->isRecordType()) {
	//if (clang::isa<clang::RecordType>(type)) {
		auto recordDecl = clang::cast<clang::RecordType>(type.getCanonicalType())->getDecl();
		std::string recordName = recordDecl->getName().str();
		if (state->ruleContext->recordSorts.count(recordName)) {
			return state->ruleContext->recordSorts.at(recordName).sort;
		}
		std::vector<std::string> fieldNames;
		std::vector<char const*> fieldNamesCstr;
		std::vector<z3::sort> fieldSorts;
		for (auto field : recordDecl->fields()) {
			fieldNames.push_back(field->getName().str());
			fieldSorts.push_back(getSort(field->getType()));
		}
		for (int i = 0; i < fieldNames.size(); i++) {
			fieldNamesCstr.push_back(fieldNames[i].c_str());
		}
		z3::func_decl_vector getters(*state->z3_ctx);
		z3::func_decl record = state->z3_ctx->tuple_sort(
			recordName.c_str(),
			(unsigned int)fieldNames.size(),
			fieldNamesCstr.data(),
			fieldSorts.data(),
			getters);
		std::unordered_map<std::string, z3::func_decl> mapGetters;
		for (int i = 0; i < fieldNames.size(); i++) {
			mapGetters.insert({ fieldNames[i], getters[i] });
		}
		z3::sort recordSort = record.range();
		state->ruleContext->recordSorts.insert({ recordName, { recordSort, mapGetters } });
		//std::cout << "insert: " << recordName << std::endl;
		return recordSort;
	}
	Log(std::cout << "[UNKNOWN_SORT]");
	return state->z3_ctx->uninterpreted_sort("unknown_sort");
	//throw AnalyzerException("Not implemented");
}

int ASTVisitor::getArraySize(clang::QualType const& type) const {
	if (type->isExtVectorType()) {
		return clang::cast<clang::ExtVectorType>(type.getDesugaredType(*astContext))->getNumElements();
	}
	if (clang::isa<clang::ConstantArrayType>(type)) {
		return static_cast<int>(clang::cast<clang::ConstantArrayType>(type)->getSize().getLimitedValue());
	}
	return -1;
}

ASTConsumer::ASTConsumer(AnalyzerContext* analyzerContext)
	: visitor(analyzerContext)
{}

bool ASTConsumer::HandleTopLevelDecl(clang::DeclGroupRef declarations) {
	for (auto decl : declarations) {
		if (!visitor.TraverseDecl(decl)) {
			return false;
		}
	}
	return true;
}
