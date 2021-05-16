__device__ void f() {
	int a[5];
	a[4] = 42;
	a[50] = 42;
}

__device__ void with_args(int i) {
	int a[5];
	a[i] = 42;
}

__device__ void with_buffer_arg(int *a, int i) {
	a[i] = 42;
}

__global__ void test_call_user_functions() {
	int a[5];
	a[4] = 42;
	a[5] = 42;
	f();
	with_args(4);
	with_args(5);
	with_buffer_arg(a, 4);
	with_buffer_arg(a, 5);
}