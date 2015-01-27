/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <math.h>
#include "numeric.h"

static const double PI = 3.14159265358979323846;

double Romberg(double (*fkt)(double), double a, double b, int N)
{
    int i;
    if (N > 14)
        N = 14;

    double T[15];

    unsigned int w = 1;
    for (i = 0; i <= N; i++)
    {
        double temp = 0.;
        double h = (b - a) / w;
        double h2 = h / 2.0;

        double f0 = fkt(a);
        for (unsigned int v = 1; v <= w; v++)
        {
            double f1 = fkt(a + v * h);
            temp += h2 * (f0 + f1);
            f0 = f1;
        }
        w *= 2;
        T[i] = temp;
    }

    double nenner = 4.0;
    for (int k = 1; k <= N; k++)
    {
        double ti1 = T[i - 1];
        for (i = k; i <= N; i++)
        {
            double ti = T[i];
            T[i] = ti + (ti - ti1) / (nenner - 1);
            ti1 = ti;
        }
        nenner *= 4.0;
    }

    return T[N];
}

double Simpson(double (*fkt)(double), double t1, double t2, int N)
{
    double H = (t2 - t1) / N;
    double h = H / 2.0;

    double v;
    int j;

    double s1 = 0;
    double s2 = 0;
    double arg = 0;

    for (j = 0; j < N; j++)
    {
        arg = t1 + (2 * j + 1) * h;
        s1 += fkt(arg);
    }

    for (j = 1; j < N; j++)
    {
        arg = t1 + (2 * j) * h;
        s2 += fkt(arg);
    }

    v = (h / 3) * (fkt(t1) + 4 * s1 + 2 * s2 + fkt(t2));

    return v;
}

static double gauss(double t)
{
    static double spi = 1.0 / sqrt(2.0 * PI);
    return spi * exp(-0.5 * t * t);
}

static double sinq(double t)
{
    return sin(t * t * 0.5 * PI);
}

static double cosq(double t)
{
    return cos(t * t * 0.5 * PI);
}

double HlFresnelSinus(double t)
{
    const int N = 22;
    static double FS[40];
    static int first = true;

    if (first)
    {
        first = false;
        FS[0] = Romberg(sinq, 0.0, 1.0, 10);
        for (int i = 1; i < N; i++)
        {
            FS[i] = FS[i - 1] + Romberg(sinq, double(i), double(i + 1), 10);
        }
    }

    int sgn = (t >= 0) ? 1 : -1;
    double at = floor(fabs(t));

    if ((at < N) && (at > 1))
    {
        int n = int(at);
        return sgn * (FS[n - 1] + Romberg(sinq, at, fabs(t), 6));
    }

    return Romberg(sinq, 0, t, 8);
}

double HlFresnelCosinus(double t)
{
    const int N = 22;
    static double FS[40];
    static int first = true;

    if (first)
    {
        first = false;
        FS[0] = Romberg(cosq, 0.0, 1.0, 10);
        for (int i = 1; i < N; i++)
        {
            FS[i] = FS[i - 1] + Romberg(cosq, double(i), double(i + 1), 10);
        }
    }

    int sgn = (t >= 0) ? 1 : -1;
    double at = floor(fabs(t));

    if ((at < N) && (at > 1))
    {
        int n = int(at);
        return sgn * (FS[n - 1] + Romberg(cosq, at, fabs(t), 6));
    }

    return Romberg(cosq, 0, t, 8);
}

double HlErf(double t)
{
    const int N = 20;
    static double FS[40];
    static int first = true;

    if (first)
    {
        first = false;
        FS[0] = Romberg(gauss, 0.0, 1.0, 10);
        for (int i = 1; i < N; i++)
        {
            FS[i] = FS[i - 1] + Romberg(gauss, double(i), double(i + 1), 10);
        }
    }

    int sgn = (t >= 0) ? 1 : -1;
    double at = floor(fabs(t));

    if ((at < N) && (at > 1))
    {
        int n = int(at);
        return sgn * (FS[n - 1] + Romberg(gauss, at, fabs(t), 7)) + 0.5;
    }

    return Romberg(gauss, 0, t, 8) + 0.5;
}

double HlSi(double t)
{
    const int N = 20;
    static double FS[40];
    static int first = true;

    if (first)
    {
        first = false;
        FS[0] = Romberg(HlSinc, 0.0, 1.0, 10);
        for (int i = 1; i < N; i++)
        {
            FS[i] = FS[i - 1] + Romberg(HlSinc, double(i), double(i + 1), 10);
        }
    }

    int sgn = (t >= 0) ? 1 : -1;
    double at = floor(fabs(t));

    if ((at < N) && (at > 1))
    {
        int n = int(at);
        return sgn * (FS[n - 1] + Romberg(HlSinc, at, fabs(t), 7));
    }

    return Romberg(HlSinc, 0, t, 8);
}

double HlSign(double z)
{
    return (z < 0) ? -1 : (z > 0) ? 1 : 0;
}

double HlSinc(double t)
{
    return (t == 0) ? 1 : sin(t) / t;
}

double HlSignum(double z, int n)
{
    if (0 == n)
        return HlSign(z);
    else
        return 0;
}

double HlBetrag(double z, int n)
{
    if (0 == n)
        return fabs(z);
    else
        return HlSignum(z, n - 1);
}
