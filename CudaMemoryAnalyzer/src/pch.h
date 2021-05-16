#pragma once

#pragma warning(disable : 4146 4702)
#pragma warning(push, 0)
#define _SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING
#include <clang/Basic/FileManager.h>
#include <clang/Basic/Builtins.h>
#include <clang/Parse/ParseAST.h>
#include <clang/Lex/Preprocessor.h>
#include <clang/Lex/PreprocessorOptions.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <clang/Basic/TargetInfo.h>
#include <clang/AST/RecursiveASTVisitor.h>
//#include <ToolChains/MSVC.h>
#include <comdef.h>
#include <ToolChains/MSVCSetupApi.h>
#include <llvm/Support/Process.h>
#include <llvm/Support/ConvertUTF.h>
#include <llvm/Support/COM.h>
#include <llvm/ADT/Optional.h>
#undef _SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING
#pragma warning(pop)
#pragma warning(default : 4146 4702)

#include <memory>
#include <iostream>