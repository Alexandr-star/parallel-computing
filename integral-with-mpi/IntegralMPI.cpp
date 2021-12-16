#include <iostream>
#include <math.h>
#include <mpi.h>

double Integral(double a, double b, double N)
{
	double step = (b - a) / N;
	double result = 0;

	for (int i = 0; i < N; i++)
	{
		double value1 = a + i * step;
		double value2 = a + (i + 1) * step;

		result += ((sin(value1) + sin(value2) * 0.5) * (value2 - value1));
	}
	return result;
}

int main(int argc, char* argv[])
{
	MPI_Init(&argc, &argv);

	int rank;
	int size;
	double buffer[2];

	MPI_Request request;
	MPI_Status status;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (rank == 0)
	{
		std::cout << "Rank " << rank << std::endl;

		double integral = Integral(rank * 7, (rank + 1) * 3, 64.5);

		std::cout << "Result integral: " << integral << std::endl;

		MPI_Send(&integral, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
	}

	if (rank == 1)
	{
		std::cout << "Rank " << rank << std::endl;

		MPI_Recv(&buffer[0], 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);

		double integral = Integral(rank * 7, (rank + 1) * 3, 64.5);

		std::cout << "Result integral: " << integral << std::endl;

		integral += buffer[0];

		std::cout << "Result integral + last result (" << buffer[0] << "): " << integral << std::endl;

		MPI_Send(&integral, 1, MPI_DOUBLE, 2, 0, MPI_COMM_WORLD);
	}

	if (rank == 2)
	{
		std::cout << "Rank " << rank << std::endl;

		MPI_Recv(&buffer[1], 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &status);

		double integral = Integral(rank * 7, (rank + 1) * 3, 64.5);

		std::cout << "Result integral: " << integral << std::endl;

		integral += buffer[1];

		std::cout << "Result integral + last result (" << buffer[1] << "): " << integral << std::endl;
	}

	MPI_Finalize();

	return 0;
}
