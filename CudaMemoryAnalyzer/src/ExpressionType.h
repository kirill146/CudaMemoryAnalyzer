#pragma once
#include <z3++.h>

struct ExpressionType {
	ExpressionType(clang::QualType type, z3::sort const& sort, int array_size = -1)
		: type(type)
		, sort(sort)
		, array_size(array_size)
	{}
public:
	clang::QualType type;
	z3::sort sort;
	int array_size;
};