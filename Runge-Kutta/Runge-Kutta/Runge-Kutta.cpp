#include <cstdio>
#include <math.h>

/// <summary>
/// Пример дифференциального уравнения y'(y + 1)sinx + 2x = y^2
/// </summary>
/// <param name="x">Параметр x</param>
/// <param name="y">Параметр y</param>
/// <returns></returns>
double F(double x, double y) 
{
    return (x + y)/2;
}

/// <summary>
/// Метод Рунге-Кутта
/// </summary>
/// <param name="step">величина шага сетки по x</param>
/// <param name="x">значение конца интервала</param>
/// <param name="x0">значение начала интервала</param>
/// <param name="y0">начальное значение</param>
/// <returns></returns>
double RungeKutta(double step, double x, double x0, double y0) {
    int numberSteps = (int)((x - x0) / step);
    double k1, k2, k3, k4;
    double y = y0;

    for (int i = 1; i <= numberSteps; i++)
    {
        k1 = step * F(x0, y);
        k2 = step * F(x0 + 0.5 * step, y + 0.5 * k1);
        k3 = step * F(x0 + 0.5 * step, y + 0.5 * k2);
        k4 = step * F(x0 + step, y + k3);

        y = y + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4);;

        x0 = x0 + step;
    }

    return y;
}

int main()
{
    double x = 1.0, x0 = 0.0, y0 = 0.1, step = 0.1;

    double result = RungeKutta(step, x, x0, y0);
    printf("%f", result);

    return 0;
}
