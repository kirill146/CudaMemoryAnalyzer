__global__ void test_uchar4(uchar4* const c)
{
	int a[5];
	uchar4 val;
	val.x = 10;
	a[val.x];
	uchar4 val2 = val;
	a[val2.x];
	uchar4 val3;
	val3 = val;
	a[val3.x];
	uchar4 val4[3];
	val4[1] = val;
	a[val4[1].x];
	c[1].y = 9;
	c[1].w = 3;
	int val5 = c[1].y;
	a[val5];
}