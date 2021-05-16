__global__ void test_restrict() {
	int a[5];
	int* __restrict__ pa = a;
	pa[0] = 1;
	a[0];
}

__global__ void test_restrict_args(int* __restrict__ a, int* __restrict__ b, int n) {
	for (int i = 0; i < n; i++) {
		a[i] = b[i];
	}
}

__global__ void test_no_restrict_violation(int* __restrict__ a, int* __restrict__ b, int* __restrict__ c, int n) {
	// b == c, but only reads happen
	for (int i = 0; i < n; i++) {
		a[i] = b[i] + c[i];
	}
}

__global__ void test_restrict_no_reads(int* __restrict__ a, int* __restrict__ b) {
	a[3] = 42;
	a[0] = 5;
	if (a[0] == 6) {
		b[2];
		b[3];
	}
}

__global__ void test_restrict_conditional_reads(int* __restrict__ a, int* __restrict__ b) {
	a[3] = 42;
	a[0] = 5;
	if (a[0] == 5) {
		b[2];
		b[3];
	}
}

__global__ void test_restrict_reads_in_for(int* __restrict__ a, int* __restrict__ b) {
	a[3] = 42;
	for (int i = 1; i < 5; i++) {
		b[i];
	}
}

__global__ void test_restrict_reads_in_while(int* __restrict__ a, int* __restrict__ b) {
	a[3] = 42;
	int i = 0;
	while (i++ < 4) {
		b[i];
	}
}

__global__ void test_restrict_read_in_condition(int* __restrict__ a, int* __restrict__ b) {
	a[3] = 42;
	if (b[3] == 42) {
	
	}
}

__global__ void test_restrict_no_writes(int* __restrict__ a, int* __restrict__ b) {
	a[3];
	a[0];
	int x = 5;
	if (x == 6) {
		b[2] = 42;
		b[3] = 42;
	}
}

__global__ void test_restrict_conditional_writes(int* __restrict__ a, int* __restrict__ b) {
	a[3];
	a[0];
	int x = 5;
	if (x == 5) {
		b[2] = 42;
		b[3] = 42;
	}
}

__global__ void test_restrict_writes_in_for(int* __restrict__ a, int* __restrict__ b) {
	a[3];
	for (int i = 1; i < 5; i++) {
		b[i] = 42;
	}
}

__global__ void test_restrict_writes_in_while(int* __restrict__ a, int* __restrict__ b) {
	a[3];
	int i = 0;
	while (i++ < 4) {
		b[i] = 42;
	}
}

__global__ void test_restrict_read_in_while_condition(int* __restrict__ a, int* __restrict__ b) {
	a[3] = 42;
	while (b[3] == 42) {
		b[3] = 0;
	}
}

__global__ void test_restrict_builtin_var_read(int* __restrict__ a, int* __restrict__ b) {
	a[3] = 42;
	b[threadIdx.x];
}

__global__ void test_restrict_builtin_var_write(int* __restrict__ a, int* __restrict__ b) {
	a[3];
	b[threadIdx.x] = 42;
}

__global__ void test_restrict_sum(int* __restrict__ a, int* __restrict__ b, int* __restrict__ c) {
    *a = 5;
    *b = 6;
    *c = *a + *b;
}

__global__ void test_restrict_self_assignment(int* __restrict__ a, int* __restrict__ b) {
	*a = *b;
}