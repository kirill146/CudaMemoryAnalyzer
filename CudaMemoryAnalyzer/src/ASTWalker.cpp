#include "pch.h"
#include "ASTWalker.h"
#include "ASTVisitor.h"
#include "AnalyzerException.h"
#include "DirectorySearch.h"
#include <stdlib.h>

std::filesystem::path PathByEnvVar(std::string const& varName) {
	size_t requiredSize;
	getenv_s(&requiredSize, nullptr, 0, varName.c_str());
	if (requiredSize == 0) {
		throw AnalyzerException("Environment variable " + varName + " is not defined");
	}
	std::vector<char> var(requiredSize);
	getenv_s(&requiredSize, var.data(), requiredSize, varName.c_str());
	return var.data();
}

std::vector<std::string> ASTWalker::findIncludes(char const* llvmIncludePath) const {
	//MSVCToolChain::ToolsetLayout VSLayout = MSVCToolChain::ToolsetLayout::OlderVS;
	std::string VCToolChainPath;
	bool msvcFound = findVCToolChain(VCToolChainPath/*, VSLayout*/);
	if (!msvcFound) {
		throw AnalyzerException("MSVC include path not found");
	}
	std::string VCIncludePath = VCToolChainPath + "\\include";
	//std::cout << "VC include path: " << VCIncludePath << std::endl;
	if (!std::filesystem::exists(VCIncludePath)) {
		throw AnalyzerException("MSVC include path not found");
	}

	std::string UniversalCRTSdkPath;
	std::string UCRTVersion;
	if (!getUniversalCRTSdkDir2(UniversalCRTSdkPath, UCRTVersion)) {
		throw AnalyzerException("ucrt include path not found");
	}
	std::string UCRTIncludePath = UniversalCRTSdkPath + "Include\\" + UCRTVersion + "\\ucrt";
	//std::cout << "UCRT include path: " << UCRTIncludePath << std::endl;
	if (!std::filesystem::exists(VCIncludePath)) {
		throw AnalyzerException("ucrt include path not found");
	}
	
	std::filesystem::path nvGpuToolkitInclude = PathByEnvVar("CUDA_PATH");
	nvGpuToolkitInclude /= "include";
	//std::cout << "nv include path: " << nvGpuToolkitInclude << std::endl;
	if (!std::filesystem::exists(nvGpuToolkitInclude)) {
		throw AnalyzerException("Nvidia Toolkit include path not found");
	}
	
	std::filesystem::path llvmInclude;
	if (strcmp(llvmIncludePath, "") != 0) {
		llvmInclude = llvmIncludePath;
	} else {
		llvmInclude = PathByEnvVar("LLVM_PROJ_DIR");
		llvmInclude /= "clang\\lib\\Headers";
	}
	//std::cout << "LLVM include path: " << llvmInclude.string() << std::endl;
	if (!std::filesystem::exists(llvmInclude)) {
		throw AnalyzerException("LLVM include path not found");
	}
	//std::cout << VCIncludePath << std::endl << UCRTIncludePath << std::endl;
	return {
		nvGpuToolkitInclude.string(),
		llvmInclude.string(),
		VCIncludePath,
		UCRTIncludePath
	};
}

ASTWalker::ASTWalker(
	AnalyzerContext* analyzerContext,
	std::vector<std::string> additionalIncludeDirs,
	char const* llvmIncludePath
)
	: compInst()
{
	auto *pTextDiagnosticPrinter = new clang::TextDiagnosticPrinter(llvm::errs(), &compInst.getDiagnosticOpts());
	compInst.createDiagnostics(pTextDiagnosticPrinter);
	
	auto* fileManager = compInst.createFileManager();

	auto& diagnosticsEngine = compInst.getDiagnostics();
	auto targetOptions = std::make_shared<clang::TargetOptions>();
	//std::cout << "defaultTargetTriple: " << llvm::sys::getDefaultTargetTriple() << std::endl;
	targetOptions->Triple = "nvptx64-nvidia-cuda";
	targetOptions->HostTriple = "x86_64-pc-windows-msvc";
	targetOptions->SDKVersion = clang::VersionTuple(10, 1);
	targetOptions->CPU = "sm_50";
	//std::cout << "triple: " << targetOptions->Triple << std::endl;
	clang::TargetInfo *pTargetInfo =
		clang::TargetInfo::CreateTargetInfo(
			diagnosticsEngine,
			targetOptions);
	
	clang::FrontendOptions frontendOptions;
	frontendOptions.AuxTriple = "x86_64-pc-windows-msvc";
	frontendOptions.AuxTargetCPU = "x86-64";
	compInst.setTarget(pTargetInfo);
	auto TO = std::make_shared<clang::TargetOptions>();
	TO->Triple = llvm::Triple::normalize(frontendOptions.AuxTriple);
	TO->HostTriple = targetOptions->Triple;
	clang::TargetInfo* newTI = clang::TargetInfo::CreateTargetInfo(diagnosticsEngine, TO);
	compInst.setAuxTarget(newTI);
	compInst.createSourceManager(*fileManager);
	
	clang::HeaderSearchOptions& hso = compInst.getHeaderSearchOpts();
	
	clang::LangOptions& languageOptions = compInst.getLangOpts();
	clang::CompilerInvocation::setLangDefaults(
		languageOptions,
		clang::InputKind(clang::Language::CUDA, clang::InputKind::Source, false),
		llvm::Triple(pTargetInfo->getTriple()),
		compInst.getPreprocessorOpts().Includes,
		//compInst.getPreprocessorOpts(),
		clang::LangStandard::lang_cxx14);

	languageOptions.CUDAAllowVariadicFunctions = 1;
	languageOptions.CUDAIsDevice = 1;
	languageOptions.MSVCCompat = 1;
	languageOptions.MicrosoftExt = 1;
	languageOptions.DeclSpecKeyword = 1;
	languageOptions.MSCompatibilityVersion = 192829334;
	languageOptions.Trigraphs = !languageOptions.GNUMode && !languageOptions.MSVCCompat && !languageOptions.CPlusPlus17;
	languageOptions.AsmBlocks = languageOptions.MicrosoftExt;
	languageOptions.MathErrno = 0;
	languageOptions.ModulesSearchAll = languageOptions.Modules;
	languageOptions.NoInlineDefine = !languageOptions.Optimize;
	languageOptions.WCharIsSigned = 1;
	/*languageOptions.CUDA = 1;
	languageOptions.CPlusPlus = 1;
	languageOptions.CPlusPlus14 = 1;
	languageOptions.Exceptions = 1;
	languageOptions.Deprecated = 1;*/
	languageOptions.MSVCCompat = 0;
	languageOptions.ObjCDefaultSynthProperties = 1;
	languageOptions.Trigraphs = 1;
	languageOptions.WChar = 1;
	languageOptions.GNUKeywords = 0;
	languageOptions.Digraphs = 1;
	languageOptions.CXXOperatorNames = 1;
	languageOptions.DoubleSquareBracketAttributes = 1;
	/*
	std::cout << "Deprecated: " << languageOptions.Deprecated << std::endl;
	std::cout << "Exceptions: " << languageOptions.Exceptions << std::endl;
	std::cout << "CPlusPlus: " << languageOptions.CPlusPlus << std::endl;
	std::cout << "CPlusPlus11: " << languageOptions.CPlusPlus11 << std::endl;
	std::cout << "CPlusPlus14: " << languageOptions.CPlusPlus14 << std::endl;
	std::cout << "CPlusPlus17: " << languageOptions.CPlusPlus17 << std::endl;
	std::cout << "CPlusPlus20: " << languageOptions.CPlusPlus20 << std::endl;
	std::cout << "HIP: " << languageOptions.HIP << std::endl;
	std::cout << "CUDA: " << languageOptions.CUDA << std::endl;
	std::cout << "CUDAIsDevice: " << languageOptions.CUDAIsDevice << std::endl;
	std::cout << "MicrosoftExt: " << languageOptions.MicrosoftExt << std::endl;
	std::cout << "MSVCCompat: " << languageOptions.MSVCCompat << std::endl;	
	*/
	std::vector<std::string> systemIncludes = findIncludes(llvmIncludePath);
	additionalIncludeDirs.insert(additionalIncludeDirs.end(), systemIncludes.begin(), systemIncludes.end());
	for (auto const& includeDir : additionalIncludeDirs) {
		hso.AddPath(includeDir, clang::frontend::IncludeDirGroup::Angled, false, false);
	}
	hso.AddSystemHeaderPrefix("__", true); // suppress some warnings on system headers
	compInst.getPreprocessorOpts().Includes.push_back("__clang_cuda_runtime_wrapper.h");
	//compInst.getPreprocessorOpts().Includes.push_back("Inputs/cuda.h"); 
	compInst.createPreprocessor(clang::TU_Complete);
	auto& PP = compInst.getPreprocessor();

	compInst.createASTContext(); 
	//std::unique_ptr<clang::ASTConsumer> cons = clang::CreateASTPrinter(NULL, "");
	compInst.setASTConsumer(std::make_unique<ASTConsumer>(analyzerContext));
	PP.getBuiltinInfo().initializeBuiltins(PP.getIdentifierTable(), PP.getLangOpts());
}

void ASTWalker::walk(std::filesystem::path const& filePath) {
	llvm::ErrorOr<clang::FileEntry const*> pFile = compInst.getFileManager().getFile(filePath.string());
	if (!pFile) {
		throw AnalyzerException("File not found: " + filePath.string());
	}
	auto& sourceManager = compInst.getSourceManager();
	auto mainID = sourceManager.getOrCreateFileID(pFile.get(), clang::SrcMgr::C_System);
	sourceManager.setMainFileID(mainID);
	
	auto& textDiagnosticPrinter = compInst.getDiagnosticClient();
	textDiagnosticPrinter.BeginSourceFile(compInst.getLangOpts(), &compInst.getPreprocessor());
	clang::ParseAST(compInst.getPreprocessor(), &compInst.getASTConsumer()/*cons.get()*/, compInst.getASTContext());
	textDiagnosticPrinter.EndSourceFile();
}

clang::SourceManager const& ASTWalker::getSourceManager() const {
	return compInst.getSourceManager();
}

clang::ASTContext const& ASTWalker::getASTContext() const {
	return compInst.getASTContext();
}
