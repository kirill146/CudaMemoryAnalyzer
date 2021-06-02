#include <iostream>
#include "json.hpp"
#include <fstream>
#include "ExecutionConfig.h"
#include "CudaBuffer.h"
#include "CudaMemoryAnalyzer.h"

dim3 parseDim3(nlohmann::json const& val) {
	dim3 res;
	if (val.is_number_unsigned()) {
		res.x = val.get<unsigned int>();
	} else if (val.is_array()) {
		auto vec = val.get<std::vector<unsigned int>>();
		if (vec.size() >= 1) {
			res.x = vec[0];
		}
		if (vec.size() >= 2) {
			res.y = vec[1];
		}
		if (vec.size() >= 3) {
			res.z = vec[2];
		}
	}
	return res;
}

int main(int argc, char *argv[]) {
	std::string inputFile = "example.json";
	if (argc >= 2) {
		inputFile = std::string(argv[1]);
	}
	if (inputFile == "example.json") {
		std::cout << "Usage:\n\tcma.exe input.json\n";
	}
	std::cout << "Input json file: " << inputFile << std::endl;
	std::ifstream fin(inputFile);
	if (!fin.is_open()) {
		std::cout << "Cannot open file" << std::endl;
		return -1;
	}
	try {
		std::vector<uint64_t> argsStorage;
		std::vector<CudaBuffer> cudaBuffersStorage;
		nlohmann::json params;
		fin >> params;
		std::string filePath = params.at("file");
		std::string kernelName = params.at("kernelName");
		std::vector<std::string> additionalIncludeDirs;
		for (auto const& dir : params.value("additionalIncludeDirs", std::vector<std::string>())) {
			additionalIncludeDirs.push_back(dir);
		}
		std::vector<void const*> args;
		nlohmann::json jsonArr = params.at("args");
		argsStorage.resize(jsonArr.size());
		cudaBuffersStorage.reserve(jsonArr.size());
		for (int i = 0; i < jsonArr.size(); i++) {
			if (jsonArr[i].is_string()) {
				std::string str = jsonArr[i];
				if (str.find('.') != std::string::npos) {
					// float or double
					if (str.find('f') != std::string::npos) {
						// float
						float tmp = std::stof(str);
						memcpy((void*)&argsStorage[i], &tmp, sizeof(float));
						args.push_back(&argsStorage[i]);
					} else {
						// double
						double tmp = std::stod(str);
						memcpy((void*)&argsStorage[i], &tmp, sizeof(double));
						args.push_back(&argsStorage[i]);
					}
				} else {
					// integer
					sscanf_s(str.c_str(), "%zu", &argsStorage[i]);
					args.push_back(&argsStorage[i]);
				}
			} else if (jsonArr[i].is_number_integer()) {
				cudaBuffersStorage.emplace_back(jsonArr[i]);
				args.push_back(cudaBuffersStorage.back().Get());
			} else {
				throw std::exception("Unknown arg type in args");
			}
		}
		std::vector<uint64_t> templateArgs;
		for (auto const& arg : params.at("templateArgs")) {
			if (arg.is_number_integer()) {
				templateArgs.push_back(arg);
			} else if (arg.is_number_float()) {
				double tmp = arg.get<double>();
				templateArgs.push_back(0);
				memcpy((void*)&templateArgs.back(), &tmp, sizeof(double));
			} else {
				throw std::exception("Unknown arg type in templateArgs");
			}
		}
		std::string outputFile;
		nlohmann::json outputFileVal = params.at("outputFile");
		if (outputFileVal.is_null()) {
			outputFile = "";
		} else {
			outputFile = outputFileVal.get<std::string>();
		}
		dim3 gridDim = parseDim3(params.at("gridDim"));
		dim3 blockDim = parseDim3(params.at("blockDim"));
		size_t dynamicMemorySize = params.at("dynamicMemorySize");
		ExecutionConfig executionConfig(gridDim, blockDim, dynamicMemorySize);
		nlohmann::json llvmIncludePathVal = params.at("llvmIncludePath");
		std::string llvmIncludePath;
		if (llvmIncludePathVal.is_null()) {
			llvmIncludePath = "";
		} else {
			llvmIncludePath = llvmIncludePathVal.get<std::string>();
		}
		/*
		std::cout << "file: " << filePath << std::endl;
		std::cout << "kernel: " << kernelName << std::endl;
		std::cout << "additional include directories: " << std::endl;
		for (std::string const& dir : additionalIncludeDirs) {
			std::cout << "\t" << dir << std::endl;
		}
		std::cout << "args cnt: " << args.size() << std::endl;
		std::cout << "template args cnt: " << templateArgs.size() << std::endl;
		std::cout << "\t";
		for (int i = 0; i < templateArgs.size(); i++) {
			std::cout << templateArgs[i] << ' ';
		}
		std::cout << std::endl;
		std::cout << "output file: ";
		if (outputFile == "") {
			std::cout << "<stdout>" << std::endl;
		} else {
			std::cout << outputFile << std::endl;
		}
		std::cout << "execution config: " << std::endl;
		std::cout << "\tgridDim: " << executionConfig.gridDim.x << ' ' << executionConfig.gridDim.y << ' ' << executionConfig.gridDim.z << std::endl;
		std::cout << "\tblockDim: " << executionConfig.blockDim.x << ' ' << executionConfig.blockDim.y << ' ' << executionConfig.blockDim.z << std::endl;
		std::cout << "\tdynamic memory size: " << executionConfig.dynamicMemorySize << std::endl;
		std::cout << "llvm include path: ";
		if (llvmIncludePath == "") {
			std::cout << "null" << std::endl;
		} else {
			std::cout << llvmIncludePath << std::endl;
		}*/
		if (params.value("checkBufferOverflows", false)) {
			std::cout << "Checking buffer overflows..." << std::endl;
			checkBufferOverflows(
				filePath,
				kernelName,
				additionalIncludeDirs,
				args,
				templateArgs,
				outputFile,
				executionConfig,
				llvmIncludePath.c_str()
			);
		}
		if (params.value("checkRestrictViolations", false)) {
			std::cout << "Checking restrict violations..." << std::endl;
			checkRestrictViolations(
				filePath,
				kernelName,
				additionalIncludeDirs,
				args,
				templateArgs,
				outputFile,
				executionConfig,
				llvmIncludePath.c_str()
			);
		}
	} catch (std::exception& e) {
		std::cout << e.what() << std::endl;
		return -1;
	}

	return 0;
}