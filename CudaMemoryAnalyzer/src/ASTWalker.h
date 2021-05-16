#pragma once
#include "AnalyzerContext.h"
#include <filesystem>

class ASTWalker {
public:
	ASTWalker(AnalyzerContext* analyzerContext, std::vector<std::string> additionalIncludeDirs);
	void walk(std::filesystem::path const& file_path);
	clang::SourceManager const& getSourceManager() const;
	clang::ASTContext const& getASTContext() const;
private:
	std::vector<std::string> findIncludes() const;
private:
	clang::CompilerInstance compInst;
};