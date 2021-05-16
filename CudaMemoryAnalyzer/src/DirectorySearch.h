#pragma once
// This is slightly changed code from clang/lib/Driver/ToolChains/MSVC.cpp
// (these functions aren't declared in the header MSVC.h)

#ifdef _WIN32
	#define WIN32_LEAN_AND_MEAN
	#define NOGDI
	#ifndef NOMINMAX
		#define NOMINMAX
	#endif
	#include <windows.h>
#endif

//using clang::driver::toolchains::MSVCToolChain;

bool findVCToolChain(std::string& VCToolChainPath);

bool findVCToolChainViaEnvironment(std::string& Path
//	, MSVCToolChain::ToolsetLayout& VSLayout
);

#ifdef _WIN32
bool readFullStringValue(HKEY hkey, const char* valueName, std::string& value);
#endif

bool getSystemRegistryString(const char* keyPath, const char* valueName,
	std::string& value, std::string* phValue);

bool getWindows10SDKVersionFromPath(const std::string& SDKPath, std::string& SDKVersion);

bool getUniversalCRTSdkDir2(std::string& Path, std::string& UCRTVersion);