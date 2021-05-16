#include "pch.h"
#include "DirectorySearch.h"

_COM_SMARTPTR_TYPEDEF(ISetupConfiguration, __uuidof(ISetupConfiguration));
_COM_SMARTPTR_TYPEDEF(ISetupConfiguration2, __uuidof(ISetupConfiguration2));
_COM_SMARTPTR_TYPEDEF(ISetupHelper, __uuidof(ISetupHelper));
_COM_SMARTPTR_TYPEDEF(IEnumSetupInstances, __uuidof(IEnumSetupInstances));
_COM_SMARTPTR_TYPEDEF(ISetupInstance, __uuidof(ISetupInstance));
_COM_SMARTPTR_TYPEDEF(ISetupInstance2, __uuidof(ISetupInstance2));

//using clang::driver::toolchains::MSVCToolChain;
using clang::isDigit;
using llvm::StringRef;

// Check various environment variables to try and find a toolchain.
bool findVCToolChainViaEnvironment(std::string &Path
//	, MSVCToolChain::ToolsetLayout &VSLayout
) {
	// These variables are typically set by vcvarsall.bat
	// when launching a developer command prompt.
	if (llvm::Optional<std::string> VCToolsInstallDir =
		llvm::sys::Process::GetEnv("VCToolsInstallDir")) {
		// This is only set by newer Visual Studios, and it leads straight to
		// the toolchain directory.
		Path = std::move(*VCToolsInstallDir);
		//VSLayout = MSVCToolChain::ToolsetLayout::VS2017OrNewer;
		return true;
	}
	if (llvm::Optional<std::string> VCInstallDir =
		llvm::sys::Process::GetEnv("VCINSTALLDIR")) {
		// If the previous variable isn't set but this one is, then we've found
		// an older Visual Studio. This variable is set by newer Visual Studios too,
		// so this check has to appear second.
		// In older Visual Studios, the VC directory is the toolchain.
		Path = std::move(*VCInstallDir);
		//VSLayout = MSVCToolChain::ToolsetLayout::OlderVS;
		return true;
	}

	// We couldn't find any VC environment variables. Let's walk through PATH and
	// see if it leads us to a VC toolchain bin directory. If it does, pick the
	// first one that we find.
	if (llvm::Optional<std::string> PathEnv =
		llvm::sys::Process::GetEnv("PATH")) {
		llvm::SmallVector<llvm::StringRef, 8> PathEntries;
		llvm::StringRef(*PathEnv).split(PathEntries, llvm::sys::EnvPathSeparator);
		for (llvm::StringRef PathEntry : PathEntries) {
			if (PathEntry.empty())
				continue;

			llvm::SmallString<256> ExeTestPath;

			// If cl.exe doesn't exist, then this definitely isn't a VC toolchain.
			ExeTestPath = PathEntry;
			llvm::sys::path::append(ExeTestPath, "cl.exe");
			if (!llvm::sys::fs::exists(ExeTestPath))
				continue;

			// cl.exe existing isn't a conclusive test for a VC toolchain; clang also
			// has a cl.exe. So let's check for link.exe too.
			ExeTestPath = PathEntry;
			llvm::sys::path::append(ExeTestPath, "link.exe");
			if (!llvm::sys::fs::exists(ExeTestPath))
				continue;

			// whatever/VC/bin --> old toolchain, VC dir is toolchain dir.
			llvm::StringRef TestPath = PathEntry;
			bool IsBin = llvm::sys::path::filename(TestPath).equals_lower("bin");
			if (!IsBin) {
				// Strip any architecture subdir like "amd64".
				TestPath = llvm::sys::path::parent_path(TestPath);
				IsBin = llvm::sys::path::filename(TestPath).equals_lower("bin");
			}
			if (IsBin) {
				llvm::StringRef ParentPath = llvm::sys::path::parent_path(TestPath);
				llvm::StringRef ParentFilename = llvm::sys::path::filename(ParentPath);
				if (ParentFilename == "VC") {
					Path = std::string(ParentPath);
					//VSLayout = MSVCToolChain::ToolsetLayout::OlderVS;
					return true;
				}
				if (ParentFilename == "x86ret" || ParentFilename == "x86chk"
					|| ParentFilename == "amd64ret" || ParentFilename == "amd64chk") {
					Path = std::string(ParentPath);
					//VSLayout = MSVCToolChain::ToolsetLayout::DevDivInternal;
					return true;
				}

			} else {
				// This could be a new (>=VS2017) toolchain. If it is, we should find
				// path components with these prefixes when walking backwards through
				// the path.
				// Note: empty strings match anything.
				llvm::StringRef ExpectedPrefixes[] = {"",     "Host",  "bin", "",
					"MSVC", "Tools", "VC"};

				auto It = llvm::sys::path::rbegin(PathEntry);
				auto End = llvm::sys::path::rend(PathEntry);
				for (llvm::StringRef Prefix : ExpectedPrefixes) {
					if (It == End)
						goto NotAToolChain;
					if (!It->startswith(Prefix))
						goto NotAToolChain;
					++It;
				}

				// We've found a new toolchain!
				// Back up 3 times (/bin/Host/arch) to get the root path.
				llvm::StringRef ToolChainPath(PathEntry);
				for (int i = 0; i < 3; ++i)
					ToolChainPath = llvm::sys::path::parent_path(ToolChainPath);

				Path = std::string(ToolChainPath);
				//VSLayout = MSVCToolChain::ToolsetLayout::VS2017OrNewer;
				return true;
			}

		NotAToolChain:
			continue;
		}
	}
	return false;
}

#ifdef _WIN32
bool readFullStringValue(HKEY hkey, const char *valueName,
	std::string &value) {
	std::wstring WideValueName;
	if (!llvm::ConvertUTF8toWide(valueName, WideValueName))
		return false;

	DWORD result = 0;
	DWORD valueSize = 0;
	DWORD type = 0;
	// First just query for the required size.
	result = RegQueryValueExW(hkey, WideValueName.c_str(), NULL, &type, NULL,
		&valueSize);
	if (result != ERROR_SUCCESS || type != REG_SZ || !valueSize)
		return false;
	std::vector<BYTE> buffer(valueSize);
	result = RegQueryValueExW(hkey, WideValueName.c_str(), NULL, NULL, &buffer[0],
		&valueSize);
	if (result == ERROR_SUCCESS) {
		std::wstring WideValue(reinterpret_cast<const wchar_t *>(buffer.data()),
			valueSize / sizeof(wchar_t));
		if (valueSize && WideValue.back() == L'\0') {
			WideValue.pop_back();
		}
		// The destination buffer must be empty as an invariant of the conversion
		// function; but this function is sometimes called in a loop that passes in
		// the same buffer, however. Simply clear it out so we can overwrite it.
		value.clear();
		return llvm::convertWideToUTF8(WideValue, value);
	}
	return false;
}
#endif

#pragma warning(disable : 4996)
/// Read registry string.
/// This also supports a means to look for high-versioned keys by use
/// of a $VERSION placeholder in the key path.
/// $VERSION in the key path is a placeholder for the version number,
/// causing the highest value path to be searched for and used.
/// I.e. "SOFTWARE\\Microsoft\\VisualStudio\\$VERSION".
/// There can be additional characters in the component.  Only the numeric
/// characters are compared.  This function only searches HKLM.
bool getSystemRegistryString(const char *keyPath, const char *valueName,
	std::string &value, std::string *phValue) {
	#ifndef _WIN32
	return false;
	#else
	HKEY hRootKey = HKEY_LOCAL_MACHINE;
	HKEY hKey = NULL;
	long lResult;
	bool returnValue = false;

	const char *placeHolder = strstr(keyPath, "$VERSION");
	std::string bestName;
	// If we have a $VERSION placeholder, do the highest-version search.
	if (placeHolder) {
		const char *keyEnd = placeHolder - 1;
		const char *nextKey = placeHolder;
		// Find end of previous key.
		while ((keyEnd > keyPath) && (*keyEnd != '\\'))
			keyEnd--;
		// Find end of key containing $VERSION.
		while (*nextKey && (*nextKey != '\\'))
			nextKey++;
		size_t partialKeyLength = keyEnd - keyPath;
		char partialKey[256];
		if (partialKeyLength >= sizeof(partialKey))
			partialKeyLength = sizeof(partialKey) - 1;
		strncpy(partialKey, keyPath, partialKeyLength);
		partialKey[partialKeyLength] = '\0';
		HKEY hTopKey = NULL;
		lResult = RegOpenKeyExA(hRootKey, partialKey, 0, KEY_READ | KEY_WOW64_32KEY,
			&hTopKey);
		if (lResult == ERROR_SUCCESS) {
			char keyName[256];
			double bestValue = 0.0;
			DWORD index, size = sizeof(keyName) - 1;
			for (index = 0; RegEnumKeyExA(hTopKey, index, keyName, &size, NULL, NULL,
				NULL, NULL) == ERROR_SUCCESS;
				index++) {
				const char *sp = keyName;
				while (*sp && !isDigit(*sp))
					sp++;
				if (!*sp)
					continue;
				const char *ep = sp + 1;
				while (*ep && (isDigit(*ep) || (*ep == '.')))
					ep++;
				char numBuf[32];
				strncpy(numBuf, sp, sizeof(numBuf) - 1);
				numBuf[sizeof(numBuf) - 1] = '\0';
				double dvalue = strtod(numBuf, NULL);
				if (dvalue > bestValue) {
					// Test that InstallDir is indeed there before keeping this index.
					// Open the chosen key path remainder.
					bestName = keyName;
					// Append rest of key.
					bestName.append(nextKey);
					lResult = RegOpenKeyExA(hTopKey, bestName.c_str(), 0,
						KEY_READ | KEY_WOW64_32KEY, &hKey);
					if (lResult == ERROR_SUCCESS) {
						if (readFullStringValue(hKey, valueName, value)) {
							bestValue = dvalue;
							if (phValue)
								*phValue = bestName;
							returnValue = true;
						}
						RegCloseKey(hKey);
					}
				}
				size = sizeof(keyName) - 1;
			}
			RegCloseKey(hTopKey);
		}
	} else {
		lResult =
			RegOpenKeyExA(hRootKey, keyPath, 0, KEY_READ | KEY_WOW64_32KEY, &hKey);
		if (lResult == ERROR_SUCCESS) {
			if (readFullStringValue(hKey, valueName, value))
				returnValue = true;
			if (phValue)
				phValue->clear();
			RegCloseKey(hKey);
		}
	}
	return returnValue;
	#endif // _WIN32
}
#pragma warning(default : 4996)

// Query the Setup Config server for installs, then pick the newest version
// and find its default VC toolchain.
// This is the preferred way to discover new Visual Studios, as they're no
// longer listed in the registry.
static bool findVCToolChainViaSetupConfig(std::string &Path) {
	// FIXME: This really should be done once in the top-level program's main
	// function, as it may have already been initialized with a different
	// threading model otherwise.
	llvm::sys::InitializeCOMRAII COM(llvm::sys::COMThreadingMode::SingleThreaded);
	HRESULT HR;

	// _com_ptr_t will throw a _com_error if a COM calls fail.
	// The LLVM coding standards forbid exception handling, so we'll have to
	// stop them from being thrown in the first place.
	// The destructor will put the regular error handler back when we leave
	// this scope.
	struct SuppressCOMErrorsRAII {
		#pragma warning (disable : 4100)
		static void __stdcall handler(HRESULT hr, IErrorInfo *perrinfo) {}
		#pragma warning (default : 4100)

		SuppressCOMErrorsRAII() { _set_com_error_handler(handler); }

		~SuppressCOMErrorsRAII() { _set_com_error_handler(_com_raise_error); }

	} COMErrorSuppressor;

	ISetupConfigurationPtr Query;
	HR = Query.CreateInstance(__uuidof(SetupConfiguration));
	if (FAILED(HR))
		return false;

	IEnumSetupInstancesPtr EnumInstances;
	HR = ISetupConfiguration2Ptr(Query)->EnumAllInstances(&EnumInstances);
	if (FAILED(HR))
		return false;

	ISetupInstancePtr Instance;
	HR = EnumInstances->Next(1, &Instance, nullptr);
	if (HR != S_OK)
		return false;

	ISetupInstancePtr NewestInstance;
	llvm::Optional<uint64_t> NewestVersionNum;
	do {
		bstr_t VersionString;
		uint64_t VersionNum;
		HR = Instance->GetInstallationVersion(VersionString.GetAddress());
		if (FAILED(HR))
			continue;
		HR = ISetupHelperPtr(Query)->ParseVersion(VersionString, &VersionNum);
		if (FAILED(HR))
			continue;
		if (!NewestVersionNum || (VersionNum > NewestVersionNum)) {
			NewestInstance = Instance;
			NewestVersionNum = VersionNum;
		}
	} while ((HR = EnumInstances->Next(1, &Instance, nullptr)) == S_OK);

	if (!NewestInstance)
		return false;

	bstr_t VCPathWide;
	HR = NewestInstance->ResolvePath(L"VC", VCPathWide.GetAddress());
	if (FAILED(HR))
		return false;

	std::string VCRootPath;
	llvm::convertWideToUTF8(std::wstring(VCPathWide), VCRootPath);

	llvm::SmallString<256> ToolsVersionFilePath(VCRootPath);
	llvm::sys::path::append(ToolsVersionFilePath, "Auxiliary", "Build",
		"Microsoft.VCToolsVersion.default.txt");

	auto ToolsVersionFile = llvm::MemoryBuffer::getFile(ToolsVersionFilePath);
	if (!ToolsVersionFile)
		return false;

	llvm::SmallString<256> ToolchainPath(VCRootPath);
	llvm::sys::path::append(ToolchainPath, "Tools", "MSVC",
		ToolsVersionFile->get()->getBuffer().rtrim());
	if (!llvm::sys::fs::is_directory(ToolchainPath))
		return false;

	Path = std::string(ToolchainPath.str());
	return true;
}


bool findVCToolChain(std::string& VCToolChainPath) {
	return findVCToolChainViaEnvironment(VCToolChainPath) ||
		findVCToolChainViaSetupConfig(VCToolChainPath);
}

// Find the most recent version of Universal CRT or Windows 10 SDK.
// vcvarsqueryregistry.bat from Visual Studio 2015 sorts entries in the include
// directory by name and uses the last one of the list.
// So we compare entry names lexicographically to find the greatest one.
bool getWindows10SDKVersionFromPath(const std::string &SDKPath,
	std::string &SDKVersion) {
	SDKVersion.clear();

	std::error_code EC;
	llvm::SmallString<128> IncludePath(SDKPath);
	llvm::sys::path::append(IncludePath, "Include");
	for (llvm::sys::fs::directory_iterator DirIt(IncludePath, EC), DirEnd;
		DirIt != DirEnd && !EC; DirIt.increment(EC)) {
		if (!llvm::sys::fs::is_directory(DirIt->path()))
			continue;
		StringRef CandidateName = llvm::sys::path::filename(DirIt->path());
		// If WDK is installed, there could be subfolders like "wdf" in the
		// "Include" directory.
		// Allow only directories which names start with "10.".
		if (!CandidateName.startswith("10."))
			continue;
		if (CandidateName > SDKVersion)
			SDKVersion = std::string(CandidateName);
	}

	return !SDKVersion.empty();
}

bool getUniversalCRTSdkDir2(std::string& Path, std::string& UCRTVersion) {
	// vcvarsqueryregistry.bat for Visual Studio 2015 queries the registry
	// for the specific key "KitsRoot10". So do we.
	if (!getSystemRegistryString(
		"SOFTWARE\\Microsoft\\Windows Kits\\Installed Roots", "KitsRoot10",
		Path, nullptr))
		return false;

	return getWindows10SDKVersionFromPath(Path, UCRTVersion);
}
