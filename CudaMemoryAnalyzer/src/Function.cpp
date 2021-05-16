#include "pch.h"
#include "Function.h"

Function::Function(std::string name, std::vector<std::unique_ptr<AtomicVariable>> args, std::unique_ptr<Statement> body)
	: name(std::move(name))
	, args(std::move(args))
	, body(std::move(body))
{}
