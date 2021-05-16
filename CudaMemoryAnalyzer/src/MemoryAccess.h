#pragma once
#include <z3++.h>

class Expression;

class MemoryAccess {
public:
	enum class Type {
		READ, WRITE
	};
	Expression const* base;
	z3::expr index;
	Type type;
	std::vector<z3::expr> predicates;
};