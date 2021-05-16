#include "pch.h"
#include "Operator.h"
#include "AnalyzerException.h"

void assignNewValue(Expression const* dst, z3::expr const& value, State* state) {
	if (auto variable = dynamic_cast<AtomicVariable const*>(dst)) {
		std::string name = variable->getName();
		auto oldVarAddrExpr = variable->z3AddressExpr(state);
		state->variableVersions[name] = state->acquireNextVersion(name);
		AtomicVariable newVariable(name, variable->getType(), variable->getAddress(), variable->getLocation());
		state->localVariablesPredicates.emplace_back(newVariable.toZ3Expr(state) == value);
		state->localVariablesPredicates.emplace_back(
			newVariable.z3AddressExpr(state) == oldVarAddrExpr
		);
	} else if (auto arraySubscript = dynamic_cast<ArraySubscriptExpression const*>(dst)) {
		Expression const* base = arraySubscript->getBase();
		z3::expr baseExpr = base->toZ3Expr(state);
		Expression const* index = arraySubscript->getIndex();
		z3::expr indexExpr = index->toZ3Expr(state);
		if (base->getType().array_size == -1) {
			return;
		}
		//std::cout << "sorts: " << baseExpr.get_sort().to_string() << ' ' << indexExpr.get_sort().to_string() << std::endl;
		assignNewValue(base, z3::store(baseExpr, indexExpr, value), state);
	} else if (auto op = dynamic_cast<UnaryOperator const*>(dst)) {
		if (op->getOp() == DEREF) {
			if (auto arr = dynamic_cast<AtomicVariable const*>(op->getArg())) {
				if (arr->getType().array_size == -1) {
					return;
				}
				assignNewValue(arr, z3::store(arr->toZ3Expr(state), 0, value), state);
			}
		} else {
			throw AnalyzerException("Unknown unary operator at dst in assignNewValue()");
		}
	} else {
		throw AnalyzerException("Unknown dst expression at assignNewValue()");
	}
}

UnaryOperator::UnaryOperator(UnaryOperation op, std::unique_ptr<Expression> arg,
	ExpressionType const& type, clang::SourceLocation const& location)
	: Expression(type, location)
	, op(op)
	, arg(std::move(arg))
{}

z3::expr UnaryOperator::toZ3Expr(State const* state) const {
	auto x = arg->toZ3Expr(state);
	switch (op) {
	case PRE_INCREMENT:
		return x + 1;
	case PRE_DECREMENT:
		return x - 1;
	case POST_INCREMENT:
	case POST_DECREMENT:
		return x;
	case NEGATE:
		return -x;
	case LNOT:
		return !x;
	case ADDR_OF:
		throw AnalyzerException("toZ3Expr() ADDR_OF unsupported");
	case DEREF:
		if (x.get_sort().is_array()) {
			return z3::select(x, 0);
		}
		throw AnalyzerException("toZ3Expr() DEREF of not arrays unsupported");
	}
	throw AnalyzerException("Unknown unary operator at toZ3Expr()");
}

void UnaryOperator::modifyState(State* state) const {
	auto x = arg->toZ3Expr(state);
	switch (op) {
	case PRE_INCREMENT:
	case POST_INCREMENT:
		assignNewValue(arg.get(), x + 1, state);
		return;
	case PRE_DECREMENT:
	case POST_DECREMENT:
		assignNewValue(arg.get(), x - 1, state);
		return;
	case NEGATE:
	case NOT:
	case LNOT:
	case REAL:
	case IMAG:
	case DEREF:
		return;
	}
	throw AnalyzerException("Unknown unary operator at modifyState()");
}

void UnaryOperator::rememberMemoryAccesses(State* state, std::vector<MemoryAccess>& memoryAccesses) const {
	if (op == DEREF) {
		if (auto arr = dynamic_cast<AtomicVariable const*>(arg.get())) {
			state->rememberMemoryAccess(arr, state->z3_ctx->int_val(0), MemoryAccess::Type::READ, memoryAccesses);
		}
	} else {
		arg->rememberMemoryAccesses(state, memoryAccesses);
	}
}

BinaryOperator::BinaryOperator(BinaryOperation op, std::unique_ptr<Expression> lhs,
	std::unique_ptr<Expression> rhs, ExpressionType const& type, clang::SourceLocation const& location)
	: Expression(type, location)
	, op(op)
	, lhs(std::move(lhs))
	, rhs(std::move(rhs))
{}

z3::expr BinaryOperator::toZ3Expr(State const* state) const {
	auto l = lhs->toZ3Expr(state);
	auto r = rhs->toZ3Expr(state);
	switch (op) {
	case ASSIGN:
		return state->z3_ctx->bool_val(true);
	case ADD:
		return l + r;
	case SUB:
		return l - r;
	case MUL:
		return l * r;
	case DIV:
		return l / r;
	case REM:
		return l - r * (l / r);
	case EQ:
		return l == r;
	case NEQ:
		return l != r;
	case LT:
		return l < r;
	case LE:
		return l <= r;
	case GT:
		return l > r;
	case GE:
		return l >= r;
	case LOR:
		return l || r;
	case LAND:
		return l && r;
	case ADD_ASSIGN:
	case SUB_ASSIGN:
	case MUL_ASSIGN:
	case DIV_ASSIGN:
	case REM_ASSIGN:
		return state->z3_ctx->bool_val(true); // rhs?
	}
	throw AnalyzerException("Unknown binary operator at toZ3Expr()");
}

void BinaryOperator::modifyState(State* state) const {
	auto l = lhs->toZ3Expr(state);
	auto r = rhs->toZ3Expr(state);
	switch (op) {
	case ASSIGN:
		assignNewValue(lhs.get(), r, state);
		return;
	case ADD_ASSIGN:
		assignNewValue(lhs.get(), l + r, state);
		return;
	case SUB_ASSIGN:
		assignNewValue(lhs.get(), l - r, state);
		return;
	case MUL_ASSIGN:
		assignNewValue(lhs.get(), l * r, state);
		return;
	case DIV_ASSIGN:
		assignNewValue(lhs.get(), l / r, state);
		return;
	case REM_ASSIGN:
		assignNewValue(lhs.get(), l - r * (l / r), state);
		return;
	case XOR_ASSIGN:
		throw AnalyzerException("XOR_ASSIGN unsupported");
		return;
	case ADD:
	case SUB:
	case MUL:
	case DIV:
	case REM:
	case SHL:
	case SHR:
	case EQ:
	case NEQ:
	case LT:
	case LE:
	case GT:
	case GE:
	case OR:
	case LOR:
	case AND:
	case LAND:
	case XOR:
		return;
	}
	throw AnalyzerException("Unknown binary operator at modifyState()");
}

void BinaryOperator::rememberMemoryAccesses(State* state, std::vector<MemoryAccess>& memoryAccesses) const {
	switch (op) {
	case ADD_ASSIGN:
	case SUB_ASSIGN:
	case MUL_ASSIGN:
	case DIV_ASSIGN:
	case REM_ASSIGN:
	case XOR_ASSIGN:
		lhs->rememberMemoryAccesses(state, memoryAccesses);
	case ASSIGN:
		if (auto arr = dynamic_cast<ArraySubscriptExpression const*>(lhs.get())) {
			state->rememberMemoryAccess(
				arr->getBase(),
				arr->getIndex()->toZ3Expr(state),
				MemoryAccess::Type::WRITE, memoryAccesses);
			arr->getIndex()->rememberMemoryAccesses(state, memoryAccesses);
		} else if (auto uop = dynamic_cast<UnaryOperator const*>(lhs.get())) {
			if (uop->getOp() == DEREF) {
				if (auto baseArr = dynamic_cast<AtomicVariable const*>(uop->getArg())) {
					state->rememberMemoryAccess(baseArr, state->z3_ctx->int_val(0),
						MemoryAccess::Type::WRITE, memoryAccesses);
				}
			}
			// don't remember writes to other variables
		}
		break;
	default:
		lhs->rememberMemoryAccesses(state, memoryAccesses);
		break;
	}
	rhs->rememberMemoryAccesses(state, memoryAccesses);
}

ConditionalOperator::ConditionalOperator(std::unique_ptr<Expression> cond,
	std::unique_ptr<Expression> then, std::unique_ptr<Expression> _else,
	ExpressionType const& type, clang::SourceLocation const& location)
	: Expression(type, location)
	, cond(std::move(cond))
	, then(std::move(then))
	, _else(std::move(_else))
{}

z3::expr ConditionalOperator::toZ3Expr(State const* state) const {
	z3::expr _cond = cond->toZ3Expr(state);
	z3::expr _true = then->toZ3Expr(state);
	z3::expr _false = _else->toZ3Expr(state);
	z3::expr tmp = state->getTmp(type.sort);
	//state->localVariablesPredicates.emplace_back(
	return	z3::ite(_cond, tmp == _true, tmp == _false);
	//);
	//return tmp;
}
