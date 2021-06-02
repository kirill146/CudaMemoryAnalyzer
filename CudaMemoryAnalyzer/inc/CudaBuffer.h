#pragma once

class CudaBuffer {
public:
	CudaBuffer(size_t size);
	void* Get() const;
	~CudaBuffer();
private:
	void* buf;
};