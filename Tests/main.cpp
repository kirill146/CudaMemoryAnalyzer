#include "CudaMemoryAnalyzer.h"
#include <fstream>
#include <iostream>
#include <string>
#include <filesystem>
#include "CudaBuffer.h"
#include <vector_types.h>
#include "AnalyzerException.h"

#ifdef _WIN32
#define PATH_SEPARATOR "\\"
#else
#define PATH_SEPARATOR "/"
#endif

#define PROJECT_ROOT "."
#define RESOURCES PROJECT_ROOT PATH_SEPARATOR "resources" PATH_SEPARATOR
#define KERNELS RESOURCES "kernels" PATH_SEPARATOR
#define OUT RESOURCES "out" PATH_SEPARATOR

char const* llvmIncludePath = "";

std::vector<std::string> readLines(std::string const& fileName) {
	std::ifstream fin(fileName);
	std::vector<std::string> res;
	std::string line;
	while (getline(fin, line)) {
		res.push_back(line);
	}
	return res;
}

void test_output(std::filesystem::path const& fileName, std::string const& kernelName,
	std::vector<void const*> const& args = {}, bool checkOverflows = true,
	dim3 gridDim = dim3(1), dim3 blockDim = dim3(1))
{ 
	std::string outputFile = std::string(OUT) + kernelName + ".out";
	std::string expectedFile = std::string(OUT) + kernelName + ".expected";
	std::cout << "File name: " << fileName << std::endl;
	std::cout << "Kernel name: " << kernelName << std::endl;
	//outputFile = "";
	if (checkOverflows) {
		checkBufferOverflows(fileName.string(), kernelName, {}, args, {}, outputFile, { gridDim, blockDim }, llvmIncludePath);
	} else {
		checkRestrictViolations(fileName.string(), kernelName, {}, args, {}, outputFile, { gridDim, blockDim }, llvmIncludePath);
	}
	std::vector<std::string> actual = readLines(outputFile);
	std::vector<std::string> expected = readLines(expectedFile);
	int i;
	for (i = 0; i < std::min(actual.size(), expected.size()); i++) {
		if (actual[i] != expected[i]) {
			std::cout << "Actual  : " << actual[i] << std::endl;
			//std::cout << actual[i].length() << std::endl;
			std::cout << "Expected: " << expected[i] << std::endl;
			//std::cout << expected[i].length() << std::endl;
		}
	}
	while (i < actual.size()) {
		std::cout << "Actual  : " << actual[i++] << std::endl;
		std::cout << "Expected: " << std::endl;
	}
	while (i < expected.size()) {
		std::cout << "Actual  : " << std::endl;
		std::cout << "Expected: " << expected[i++] << std::endl;
	}
	std::cout << "Test finished" << std::endl;
	std::cout << "------------------------------------------" << std::endl;
}

void test_arrays() {
	std::string fileName = KERNELS "test_arrays.cu";
	test_output(fileName, "test_arrays");
}

void test_if() {
	std::string fileName = KERNELS "test_if.cu";
	test_output(fileName, "test_if");
}

void test_loops() {
	std::string fileName = KERNELS "test_loops.cu";
	test_output(fileName, "test_while");
	test_output(fileName, "test_large_while");
	test_output(fileName, "test_statements_in_while_body");
	test_output(fileName, "test_for");
}

void test_call_functions() {
	std::string fileName = KERNELS "test_call_functions.cu";
	test_output(fileName, "test_call_user_functions");
}

void test_builtin_variables() {
	std::string fileName = KERNELS "test_builtin_variables.cu";
	test_output(fileName, "test_builtin_variables", {}, true, dim3(2, 3, 4), dim3(5, 6, 7));
}

void test_input_args() {
	CudaBuffer buffer_arg(2 * sizeof(int));
	std::vector<void const*> args;
	args.push_back(buffer_arg.Get());
	int x = 1;
	args.push_back(&x);

	std::string fileName = KERNELS "test_input_args.cu";
	test_output(fileName, "test_input_args", args);
}

void test_restrict_args() {
	int const sz = 10;
	CudaBuffer a(sz * sizeof(int));
	int n = 5;
	std::vector<void const*> args;
	args.push_back((int*)a.Get() + 1);
	args.push_back(a.Get());
	args.push_back(&n);

	std::string fileName = KERNELS "test_restrict.cu";
	test_output(fileName, "test_restrict_args", args, false);
}

void test_no_restrict_violation() {
	int const sz = 10;
	CudaBuffer a(sz * sizeof(int));
	CudaBuffer b(sz * sizeof(int));
	int n = 5;
	std::vector<void const*> args;
	args.push_back(a.Get());
	args.push_back(b.Get());
	args.push_back(b.Get());
	args.push_back(&n);

	std::string fileName = KERNELS "test_restrict.cu";
	test_output(fileName, "test_no_restrict_violation", args, false);
}

void test_restrict() {
	std::string fileName = KERNELS "test_restrict.cu";
	test_output(fileName, "test_restrict", {}, false);
}

void test_restrict_self_assignment() {
	int const sz = 10;
	CudaBuffer a(sz * sizeof(int));
	std::vector<void const*> args;
	args.push_back((int*)a.Get());
	args.push_back(a.Get());

	std::string fileName = KERNELS "test_restrict.cu";
	//checkRestrictViolations(fileName, "test_restrict_self_assignment", {}, args, {}, "", { dim3(1), dim3(1) });
	test_output(fileName, "test_restrict_self_assignment", args, false);
}

void test_restrict_accesses(std::string const& kernelName, dim3 gridDim = dim3(1), dim3 blockDim = dim3(1)) {
	int const sz = 10;
	CudaBuffer a(sz * sizeof(int));
	std::vector<void const*> args;
	args.push_back(a.Get());
	args.push_back(a.Get());

	std::string fileName = KERNELS "test_restrict.cu";
	//checkRestrictViolations(fileName, kernelName, {}, processed_arg_sizes, args, {}, "", { gridDim, blockDim });
	test_output(fileName, kernelName, args, false, gridDim, blockDim);
}

void test_restrict_sum() {
	CudaBuffer a(sizeof(int));
	CudaBuffer c(sizeof(int));
	std::vector<void const*> args;
	args.push_back(a.Get());
	args.push_back(a.Get());
	args.push_back(c.Get());

	std::string fileName = KERNELS "test_restrict.cu";
	//checkRestrictViolations(fileName, "test_restrict_sum", {}, {}, args, {}, "", { dim3(1), dim3(1) });
	test_output(fileName, "test_restrict_sum", args, false);
}

void test_restrict_accesses() {
	test_restrict_accesses("test_restrict_no_reads");
	test_restrict_accesses("test_restrict_conditional_reads");
	test_restrict_accesses("test_restrict_reads_in_for");
	test_restrict_accesses("test_restrict_reads_in_while");
	test_restrict_accesses("test_restrict_read_in_condition");
	test_restrict_accesses("test_restrict_no_writes");
	test_restrict_accesses("test_restrict_conditional_writes");
	test_restrict_accesses("test_restrict_writes_in_for");
	test_restrict_accesses("test_restrict_writes_in_while");
	test_restrict_accesses("test_restrict_read_in_while_condition");
	test_restrict_accesses("test_restrict_builtin_var_read", dim3(2, 3, 4), dim3(5, 6, 7));
	test_restrict_accesses("test_restrict_builtin_var_write", dim3(2, 3, 4), dim3(5, 6, 7));
}

void test_template() {
	checkBufferOverflows(
		"kernel.cu",
		"ggg",
		{},
		{},
		{ 40, 55 },
		"",
		{ 1, 1 }
	);
}

void test_float() {
	CudaBuffer a(4 * sizeof(float));
	float x = 3.0f;
	checkBufferOverflows(
		"kernel.cu",
		"test_float",
		{},
		{ &x },
		{},
		"",
		{ 1, 1 });
}

void run_tests() {
	test_restrict_sum();
	test_restrict_accesses();
	test_restrict();
	test_restrict_args();
	test_restrict_self_assignment();
	test_no_restrict_violation();
	test_arrays();
	test_if();
	test_call_functions();
	test_input_args();
	test_builtin_variables();
	test_loops();
}

int main(int argc, char *argv[]) {
	if (argc >= 2) {
		llvmIncludePath = argv[1];
	}
	try {
		//test_float();
		//test_template();
		run_tests();
	} catch (AnalyzerException& e) {
		std::cout << "AnalyzerException: " << e.what() << std::endl;
	} catch (std::exception& e) {
		std::cout << e.what() << std::endl;
		return -1;
	} catch(...) {
		std::cout << "Unknown exception was thrown" << std::endl;
		return -1;
	}
	return 0;
}