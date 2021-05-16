#include "CudaBuffer.h"
#include <cuda.h>
#include "AnalyzerException.h"

CudaBuffer::CudaBuffer(size_t size)
    : buf(nullptr)
{
    cudaError_t err = cudaMalloc(&buf, size);
    if (err != CUDA_SUCCESS) {
        throw AnalyzerException("Cannot allocate device memory");
    }
}

void* CudaBuffer::Get() const {
    return buf;
}

CudaBuffer::~CudaBuffer() {
    cudaFree(buf);
}
