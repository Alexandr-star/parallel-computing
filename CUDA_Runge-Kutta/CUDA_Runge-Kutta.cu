
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
using namespace std;


#define BLOCK_SIZE 256
#define SYSTEM_SIZE BLOCK_SIZE * 3
#define PARAMS_SIZE 3
#define steps BLOCK_SIZE * 10 * 3

double t0 = 0.0;
double tn = 5.0;


void SetInitParam(double* Y, double* T, double* paramsF, double t0)
{
    for (int i = 0; i < SYSTEM_SIZE; i++)
    {
        Y[i] = paramsF[i % PARAMS_SIZE]; //
    }
    T[0] = t0;
}

// <summary>
///  Система дифференциальных уравнений.
/// </summary>
/// <param name="LF">Массив уравнений</param>
/// <param name="paramsF">Массив параметров x, y, z</param>
/// <returns></returns>
__global__ void F(double a, double b, double c, double t, double* paramsF, double* Kn)
{
    int tID = blockDim.x * blockIdx.x + threadIdx.x;
    int size = SYSTEM_SIZE / BLOCK_SIZE;

    for (size_t i = tID * size; i < (tID+3) * size; i += 3)
    {
        Kn[i] = (a + t) * (paramsF[1] - paramsF[0]);
        Kn[i + 1] = paramsF[0] * (b + 1 - paramsF[2]) - paramsF[1];
        Kn[i + 2] = paramsF[0] * paramsF[1] - (c + 1) * paramsF[2];
    }
}



/// <summary>
/// Метод Рунге-Кутта
/// </summary>
/// <param name="step">величина шага сетки по x</param>
/// <param name="x">значение конца интервала</param>
/// <param name="x0">значение начала интервала</param>
/// <param name="y0">начальное значение</param>
/// <returns></returns>
__global__ void RungeKutta(double** Y, double* T, double* paramsF, double* K1, double* K2, double* K3, double* K4, double* medValues) {
    
    double h = 0.01;
    double a = 10.0;
    double b = 28.0;
    double c = 2.6;

    int tID = blockDim.x * blockIdx.x + threadIdx.x;

    int N = BLOCK_SIZE * 100;
    int threadsPerBlock = BLOCK_SIZE; int blocksPerGrid = N / threadsPerBlock;

    int size_steps = steps / BLOCK_SIZE;
    int size_sys = SYSTEM_SIZE / BLOCK_SIZE;

    for (int t = tID *size_steps; t < (tID + 1) * size_steps; t++)
    {

        F<<<threadsPerBlock, blocksPerGrid >>>(a, b, c, T[t - 1], Y[t - 1], K1);

        for (int i = tID * size_sys; i < (tID + 1) * size_sys; i++)
            medValues[i] = Y[t - 1][i] + K1[i] * (h / 2);

        F<<<threadsPerBlock, blocksPerGrid >>>(a, b, c, T[t - 1] + h / 2, medValues, K2);

        for (int i = tID * size_sys; i < (tID + 1) * size_sys; i++)
            medValues[i] = Y[t - 1][i] + K2[i] * (h / 2);

        F<<<threadsPerBlock, blocksPerGrid >>>(a, b, c, T[t - 1] + h / 2, medValues, K3);

        for (int i = tID * size_sys; i < (tID + 1) * size_sys; i++)
            medValues[i] = Y[t - 1][i] + K3[i] * h;

        F<<<threadsPerBlock, blocksPerGrid >>>(a, b, c, T[t - 1] + h, medValues, K4);

        for (int i = tID * size_sys; i < (tID + 1) * size_sys; i++)
            Y[t][i] = Y[t - 1][i] + (K1[i] + 2 * K2[i] + 2 * K3[i] + K4[i]) * h / 6;

        T[t] = T[t - 1] + h;
    }
}



int main()
{
    int N = BLOCK_SIZE * 100;

    size_t size_params = PARAMS_SIZE * sizeof(double);
    size_t size_system = SYSTEM_SIZE * sizeof(double);
    size_t size_steps = steps * sizeof(double);

    double *d_paramsF, *d_T, *d_K1, *d_K2, *d_K3, *d_K4, *d_medValues, **d_Y;

    double* h_paramsF = (double*)malloc(size_params);
    double* h_T = (double*)malloc(size_steps);
    double** h_Y = (double**)malloc(size_steps);
    double* h_K1 = (double*)malloc(size_system);
    double* h_K2 = (double*)malloc(size_system);
    double* h_K3 = (double*)malloc(size_system);
    double* h_K4 = (double*)malloc(size_system);
    double* h_medValues = (double*)malloc(size_system);

    for (int i = 0; i < steps; i++)
    {
        h_Y[i] = (double*)malloc(size_system);
    }

    h_paramsF[0] = 10.0;
    h_paramsF[1] = 10.0;
    h_paramsF[2] = 10.0;

    SetInitParam(h_Y[0], h_T, h_paramsF, t0);

    cudaMalloc((void**)&d_paramsF, size_params);
    cudaMalloc((void**)&d_T, size_steps);
    cudaMalloc((void**)&d_K1, size_system);
    cudaMalloc((void**)&d_K2, size_system);
    cudaMalloc((void**)&d_K3, size_system);
    cudaMalloc((void**)&d_K4, size_system);
    cudaMalloc((void**)&d_medValues, size_system);
    cudaMalloc((void**)&d_Y, size_steps);

    cudaMemcpy(d_paramsF, h_paramsF, size_params, cudaMemcpyHostToDevice);
    cudaMemcpy(d_T, h_T, size_steps, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, h_Y, size_steps, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K1, h_K1, size_system, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K2, h_K2, size_system, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K3, h_K3, size_system, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K4, h_K4, size_system, cudaMemcpyHostToDevice);
    cudaMemcpy(d_medValues, h_medValues, size_system, cudaMemcpyHostToDevice);

    int threadsPerBlock = BLOCK_SIZE; int blocksPerGrid = N / threadsPerBlock;

    RungeKutta<<<threadsPerBlock, blocksPerGrid >>>(d_Y, d_T, d_paramsF, d_K1, d_K2, d_K3, d_K4, d_medValues);

    double** result = (double**)malloc(size_steps);
    double* t = (double*)malloc(size_steps);

    cudaMemcpy(&result, d_Y, size_steps, cudaMemcpyDeviceToHost);
    cudaMemcpy(&t, d_T, size_steps, cudaMemcpyDeviceToHost);

    cudaFree(d_paramsF);
    cudaFree(d_T);
    cudaFree(d_Y);
    cudaFree(d_K1);
    cudaFree(d_K2);
    cudaFree(d_K3);
    cudaFree(d_K4);
    cudaFree(d_medValues);

    return 0;
}