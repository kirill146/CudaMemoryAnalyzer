#include "pch.h"
#include "CudaBuffer.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdexcept>

CudaBuffer::CudaBuffer(size_t size)
    : buf(nullptr)
{
    cudaError_t err = cudaMalloc(&buf, size);
    if (err != CUDA_SUCCESS) {
        throw std::exception("Cannot allocate device memory");
    }
}

void* CudaBuffer::Get() const {
    return buf;
}

CudaBuffer::~CudaBuffer() {
    cudaFree(buf);
}