#include <iostream>
#include "AnalyzerException.h"
#include <cuda.h>

size_t BufSizeByAddress(void const* p) {
	CUdeviceptr pbase;
	size_t psize;
	CUresult res = cuMemGetAddressRange(&pbase, &psize, (CUdeviceptr)p);
	if (res != CUDA_SUCCESS) {
		std::cout << "cuMemGetAddressRange() failed\n";
		throw AnalyzerException("Invalid pointer to buffer in kernel call");
	}
	uint64_t offset = (uint64_t)p - pbase;
	return psize - offset;
}
