#pragma once
#include <exception>
#include <string>

class AnalyzerException : public std::exception {
public:
	explicit AnalyzerException(std::string msg)
		: std::exception()
		, msg(std::move(msg))
	{}
	char const* what() const override {
		return msg.c_str();
	}
private:
	std::string msg;
};

#ifdef PEDANTIC_DEBUG
#define PEDANTIC_THROW(s) throw AnalyzerException(s)
#else
#define PEDANTIC_THROW(s)
#endif