#pragma once

#ifdef _WIN32
	#define WIN32_LEAN_AND_MEAN
	#define NOGDI
	#ifndef NOMINMAX
		#define NOMINMAX
	#endif
	#include <windows.h>
#endif

bool findVCToolChain(std::string& VCToolChainPath);

bool findVCToolChainViaEnvironment(std::string& Path);

#ifdef _WIN32
bool readFullStringValue(HKEY hkey, const char* valueName, std::string& value);
#endif

bool getSystemRegistryString(const char* keyPath, const char* valueName,
	std::string& value, std::string* phValue);

bool getWindows10SDKVersionFromPath(const std::string& SDKPath, std::string& SDKVersion);

bool getUniversalCRTSdkDir2(std::string& Path, std::string& UCRTVersion);