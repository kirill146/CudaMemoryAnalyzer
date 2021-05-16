__global__ void test_builtin_variables() {
	// gridDim ==  { 2, 3, 4 }
	// blockDim == { 5, 6, 7 }
	int a1[1];
	int a2[2];
	int a3[3];
	int a4[4];
	int a5[5];
	int a6[6];
	int a7[7];
	int a8[8];
	
	a3[gridDim.x] = 42;
	a2[gridDim.x] = 42;
	a4[gridDim.y] = 42;
	a3[gridDim.y] = 42;
	a5[gridDim.z] = 42;
	a4[gridDim.z] = 42;
	
	a6[blockDim.x] = 42;
	a5[blockDim.x] = 42;
	a7[blockDim.y] = 42;
	a6[blockDim.y] = 42;
	a8[blockDim.z] = 42;
	a7[blockDim.z] = 42;
	
	a2[blockIdx.x] = 42;
	a1[blockIdx.x] = 42;
	a3[blockIdx.y] = 42;
	a2[blockIdx.y] = 42;
	a4[blockIdx.z] = 42;
	a3[blockIdx.z] = 42;
	
	a5[threadIdx.x] = 42;
	a4[threadIdx.x] = 42;
	a6[threadIdx.y] = 42;
	a5[threadIdx.y] = 42;
	a7[threadIdx.z] = 42;
	a6[threadIdx.z] = 42;
	
	a7[gridDim.x - 1] = 42;
	a7[blockDim.x - 1] = 42;
	a7[blockIdx.x - 1] = 42;
	a7[threadIdx.x - 1] = 42;
	
	a7[gridDim.y - 1] = 42;
	a7[blockDim.y - 1] = 42;
	a7[blockIdx.y - 1] = 42;
	a7[threadIdx.y - 1] = 42;
	
	a7[gridDim.z - 1] = 42;
	a7[blockDim.z - 1] = 42;
	a7[blockIdx.z - 1] = 42;
	a7[threadIdx.z - 1] = 42;
	
	int a[10];
	int id = blockIdx.x * blockDim.x + threadIdx.x; // 0 <= id <= 9
	a[id] = 42;
	a[id + 1] = 42;
	int b[18];
	b[blockIdx.y * blockDim.y + threadIdx.y] = 42; // 0 <= index <= 17
	b[blockIdx.y * blockDim.y + threadIdx.y + 1] = 42;
}