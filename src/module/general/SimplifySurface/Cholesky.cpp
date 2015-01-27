/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <math.h>

bool
choldc(double **a, int n, double p[])
{
    int i, j, k;
    double sum;

    for (i = 0; i < n; i++)
    {
        for (j = i; j < n; j++)
        {
            for (sum = a[i][j], k = i - 1; k >= 0; k--)
                sum -= a[i][k] * a[j][k];
            if (i == j)
            {
                if (sum <= 0.0)
                    return false;
                p[i] = sqrt(sum);
            }
            else
                a[j][i] = sum / p[i];
        }
    }
    return true;
}

void
cholsl(double **a, int n, double p[], const double b[], double x[])
{
    int i, k;
    double sum;

    for (i = 0; i < n; i++)
    {
        for (sum = b[i], k = i - 1; k >= 0; k--)
            sum -= a[i][k] * x[k];
        x[i] = sum / p[i];
    }
    for (i = n - 1; i >= 0; i--)
    {
        for (sum = x[i], k = i + 1; k < n; k++)
            sum -= a[k][i] * x[k];
        x[i] = sum / p[i];
    }
}
