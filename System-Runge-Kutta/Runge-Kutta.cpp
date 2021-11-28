#include <cstdio>
#include <iostream>
#include <math.h>
#include <fstream>
#include <omp.h>


using namespace std;

double a;
double b;
double c;
double tn;
double t0;
int steps;
int threadCount;

const int sizeLF = 3;
const int sizeParams = 3;

void InputData(double *paramsF)
{
    cout << "Input param a, b, c: ";
    cin >> a;
    cin >> b;
    cin >> c;

    cout << "Input param t0, x0, y0, z0: ";
    cin >> t0;
    cin >> paramsF[1];
    cin >> paramsF[2];
    cin >> paramsF[3];

    cout << "Input tn: ";
    cin >> tn;

    cout << "Input number points: ";
    cin >> steps;

    cout << "Input thread count: ";
    cin >> threadCount;
}

void SetInitParam(double *Y, double *T, double *paramsF)
{
    for (int i = 0; i < sizeLF; i++)
    {
        Y[i] = paramsF[i];
    }
    T[0] = t0;
}

/// <summary>
///  Система дифференциальных уравнений Лоренца.
/// </summary>
/// <param name="LF">Массив уравнений</param>
/// <param name="paramsF">Массив параметров x, y, z для системы Лоренца</param>
/// <returns></returns>
void LorentzF(double t, double *paramsF, double *Kn)
{
    Kn[0] = a * (paramsF[1] - paramsF[0]);
    Kn[1] = paramsF[0] * (b - paramsF[2]) - paramsF[1];
    Kn[2] = paramsF[0] * paramsF[1] - c * paramsF[2];
}



/// <summary>
/// Метод Рунге-Кутта
/// </summary>
/// <param name="step">величина шага сетки по x</param>
/// <param name="x">значение конца интервала</param>
/// <param name="x0">значение начала интервала</param>
/// <param name="y0">начальное значение</param>
/// <returns></returns>
int RungeKutta(double **Y, double *T, double *paramsF) {
    double *K1 = (double*)malloc(sizeLF * sizeof(double));
    double *K2 = (double*)malloc(sizeLF * sizeof(double));
    double *K3 = (double*)malloc(sizeLF * sizeof(double));
    double *K4 = (double*)malloc(sizeLF * sizeof(double));
    double *medValues = (double*)malloc(sizeLF * sizeof(double));
    double h = 0.01;

    SetInitParam(Y[0], T, paramsF);

    if (medValues && K1 && K2 && K3 && K4)
    {
#pragma omp parallel for num_threads(threadCount)shared(Y)
        for (int t = 1; t < steps; t++)
        {
            LorentzF(T[t - 1], Y[t - 1], K1);

            for (int i = 0; i < sizeLF; i++)
                medValues[i] = Y[t - 1][i] + K1[i] * (h / 2);

            LorentzF(T[t - 1] + h / 2, medValues, K2);

            for (int i = 0; i < sizeLF; i++)
                medValues[i] = Y[t - 1][i] + K2[i] * (h / 2);

            LorentzF(T[t - 1] + h / 2, medValues, K3);

            for (int i = 0; i < sizeLF; i++)
                medValues[i] = Y[t - 1][i] + K3[i] * h;

            LorentzF(T[t - 1] + h, medValues, K4);

            for (int i = 0; i < sizeLF; i++)
                Y[t][i] = Y[t - 1][i] + (K1[i] + 2 * K2[i] + 2 * K3[i] + K4[i])* h/6;

            T[t] = T[t - 1] + h;
        }

        free(K1);
        free(K2);
        free(K3);
        free(K4);
        free(medValues);
    }
    else
    {
        free(K1);
        free(K2);
        free(K3);
        free(K4);
        free(medValues);
        cout << "Error null point:";
        return 1;
    }

    return 0;
}

void StreamInFile(double **Y, double *T)
{
    ofstream file;
    file.open("Lorentz.txt", ofstream::out | ofstream::trunc);
    for (int i = 0; i < steps; i++) {
        file << T[i] << " ";
    }
    file << endl;
    for (int i = 0; i < sizeLF; i++) {
        for (int j = 0; j < steps; j++)
        {
            file << Y[j][i] << " ";
        }
        file << endl;
    }
    
    file.close();
}

int main()
{
    // InputData(paramsF);

    t0 = 0.0;
    tn = 5.0;
    steps = 3000;
    a = 10.0;
    b = 28.0;
    c = 2.6;
    threadCount = 5;

    double *paramsF = (double*)malloc(sizeParams * sizeof(double));
    double *T = (double*)malloc(steps * sizeof(double));
    double **Y = (double**)malloc(steps * sizeof(double));

    if (paramsF && Y && T)
    {
        for (int i = 0; i < steps; i++)
        {
            Y[i] = (double*)malloc(sizeLF * sizeof(double));
        }


        paramsF[0] = 10.0;
        paramsF[1] = 10.0;
        paramsF[2] = 10.0;

        RungeKutta(Y, T, paramsF);
        StreamInFile(Y, T);

        free(paramsF);
        for (int i = 0; i < steps; i++)
        {
            free(Y[i]);
        }
        free(Y);
        free(T);
    }
    else
    {
        free(paramsF);
        free(Y);
        free(T);
        cout << "Error null point:";
        return 1;
    }

    return 0;
}
