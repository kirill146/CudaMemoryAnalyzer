__global__ void test_if()
{
	int a[5];
	int x = 4;
	int y = 5;

	if (x < 5) {
		a[x] = 42;
		a[x + 1] = 42;
		y = 0;
		++x;
	}
	a[y] = 42;
	a[x] = 42;

	if (x < 5) // unreachable
	{
		int z = -1;
		int local_var = 0;
		a[z] = 42; // Okay, because unreachable
	} else
	{
		a[x] = 42;
	}

	int local_var;
	a[local_var] = 42;
	if (local_var < 5)
	{
		int z = 4;
		a[z + 1] = 42;
		a[local_var] = 42;
		if (local_var >= 0)
		{
			a[local_var] = 42;
		}
	}
}