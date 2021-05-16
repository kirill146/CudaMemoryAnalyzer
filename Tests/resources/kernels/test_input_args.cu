__global__ void test_input_args(int* buffer_arg, int x) {
	int val = buffer_arg[1];
	buffer_arg[2] = 42;
	buffer_arg[x] = 42;
	buffer_arg[x + 1] = 42;
	x = 1;
	buffer_arg[x] = 42;
}