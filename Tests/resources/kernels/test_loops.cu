__global__ void test_while()
{
	int a[5];
	int x = 0;
	int i = 0;
	while (i++ < 5)
	{
		// i == 1..5
		// x == 1..5
		++x;
		a[x] = 42;
		a[x - 1] = 42;
	}
	// i == 6, x == 5
	a[i] = 42;
	a[x] = 42;
	a[x - 1] = 42;
}

__global__ void test_large_while()
{
	const int size = 10000;
	int b[size];
	int i = 0;
	int x = 0;
	while (i++ < size)
	{
		++x;
	}
	b[x] = 42;
	b[x - 1] = 42;
}

__global__ void test_statements_in_while_body()
{
	int a[5];
	int c[100];
	int out_bound = 20;
	int inner_bound = 5;
	int i = 0;
	int x = 0;
	while (i++ < out_bound)
	{
		if (i < 5)
		{
			a[i] = 42;
		}
		a[i] = 42;
		int j = 0;
		while (j++ < inner_bound)
		{
			++x;
		}
	}
	c[x] = 42;
	c[x - 1] = 42;
}

__global__ void test_for()
{
	int a[5];
	int x = 0;
	for (int i = 0; i < 5; ++i)
	{
		++x;
		a[x] = 42;
		a[x - 1] = 42;
		for (int j = 0; j < 10; ++j)
		{
			a[j] = 42;
			a[j / 2] = 42;
		}
		a[i] = 42;
		a[i + 1] = 42;
	}
	a[x] = 42;
	a[x - 1] = 42;
}