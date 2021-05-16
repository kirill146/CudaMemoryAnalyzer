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