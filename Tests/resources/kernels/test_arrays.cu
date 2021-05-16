__global__ void test_arrays() {
	int a[5];

    a[4] = 42;
    a[5] = 42;
    a[2 + 3] = 42;
    a['a'] = 42;
    a[0] = a[5] + a[6];
    int y = -1;
    a[y] = 42;
    int z;
    a[z] = 42;
    int p;
    p = 1;
    a[p] = 42;

    int b[5][6];
    b[4][5] = 42;
    b[5][5] = 42;
    b[4][6] = 42;
}