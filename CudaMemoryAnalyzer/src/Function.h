#pragma once
#include "Expression.h"

class Function {
public:
	Function(std::string name, std::vector<std::unique_ptr<AtomicVariable>> args,
		std::unique_ptr<Statement> body);
	std::vector<std::unique_ptr<AtomicVariable>> const& getArgs() const { return args; }
	Statement const* getBody() const { return body.get(); }
private:
	std::string name;
	std::vector<std::unique_ptr<AtomicVariable>> args;
	std::unique_ptr<Statement> body;
};