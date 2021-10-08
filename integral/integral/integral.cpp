#include <iostream>
#include <omp.h>
#include <math.h>

int main(int argc, char* argv[])
{
	double a = 1.0, b = 152.0;
	int N = 100000;
	double step = (b - a) / N;

	double *valueInX = (double*)malloc((N + 1) * sizeof(double));

	double itime, ftime, exec_time;
	itime = omp_get_wtime();

	if (valueInX)
	{
		for (int i = 0; i < N; i++)
		{
			valueInX[i] = a + i * step;
		}
		valueInX[N] = b;
	}
	else
	{
		printf("warning C6011 : dereferencing NULL pointer <name>");
		return 1;
	}

	double result = 0.0;

#pragma omp parallel for shared(valueInX) reduction(+:result)
	for (int i = 0; i < N; i++)
	{
		result += ((sin(valueInX[i]) + sin(valueInX[i + 1])) * 0.5) * (valueInX[i + 1] - valueInX[i]);
	}

	free(valueInX);

	ftime = omp_get_wtime();
	exec_time = ftime - itime;
	printf("Time taken is %f \n", exec_time);

	printf("integral sin(x), a = %.1f, b =  %.1f : %f", a, b, result);

	return 0;
}