/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "gaalet.h"
#include <sys/time.h>
#include <cmath>

typedef gaalet::algebra<gaalet::signature<3, 0> > em;

int main()
{
    timeval start, end;
    double solveTime;

    em::mv<1, 2, 4, 7>::type a = { 1.0, 1.0, 0.0, 0.0 };
    em::mv<0, 3, 5, 6>::type b = { cos(-M_PI * 0.25), sin(-M_PI * 0.25), 0.0, 0.0 };

    gettimeofday(&start, 0);
    for (int i = 0; i < 1e8; ++i)
    {
        //a = (~b)*b*a*(~b)*b;
        a = b * a * (~b);
    }
    gettimeofday(&end, 0);
    solveTime = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec) * 1e-6;

    std::cout << "a: " << a << std::endl;
    std::cout << "b: " << b << std::endl;
    std::cout << "operator=(): multiply solve time: " << solveTime << std::endl;

    em::mv<1, 2, 4, 7>::type d = { 1.0, 1.0, 0.0, 0.0 };
    em::mv<0, 3, 5, 6>::type e = { cos(-M_PI * 0.25), sin(-M_PI * 0.25), 0.0, 0.0 };

    gettimeofday(&start, 0);
    for (int i = 0; i < 1e8; ++i)
    {
        //d = eval((~e)*e*d*(~e)*e);
        d = eval(e * d * (~e));
    }
    gettimeofday(&end, 0);
    solveTime = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec) * 1e-6;

    std::cout << "d: " << d << std::endl;
    std::cout << "e: " << e << std::endl;
    std::cout << "eval(): multiply solve time: " << solveTime << std::endl;

    em::mv<1, 2, 4, 7>::type f = { 1.0, 1.0, 0.0, 0.0 };
    em::mv<0, 3, 5, 6>::type g = { cos(-M_PI * 0.25), sin(-M_PI * 0.25), 0.0, 0.0 };

    gettimeofday(&start, 0);
    for (int i = 0; i < 1e8; ++i)
    {
        double f0 = -g[1] * g[3] * f[2] - g[1] * g[3] * f[2] + g[2] * g[3] * f[1] + g[2] * g[3] * f[1] + g[3] * g[3] * f[3] - g[1] * g[2] * f[3] - g[1] * g[2] * f[3] + g[3] * g[3] * f[0] + g[2] * g[2] * f[0] + g[1] * g[1] * f[0] + g[0] * g[0] * f[0];
        double f1 = g[1] * g[2] * f[2] + g[1] * g[2] * f[2] + g[3] * g[3] * f[1] + g[2] * g[2] * f[1] + g[1] * g[1] * f[1] + g[0] * g[0] * f[1] - g[1] * g[3] * f[3] - g[1] * g[3] * f[3] - g[2] * g[3] * f[0] - g[2] * g[3] * f[0];
        double f2 = g[3] * g[3] * f[2] + g[1] * g[1] * f[2] + g[0] * g[0] * f[2] - g[1] * g[2] * f[1] - g[1] * g[2] * f[1] - g[2] * g[3] * f[3] - g[2] * g[3] * f[3] + g[1] * g[3] * f[0] + g[1] * g[3] * f[0];
        double f3 = g[2] * g[3] * f[2] + g[2] * g[3] * f[2] + g[2] * g[2] * f[2] + g[1] * g[3] * f[1] + g[1] * g[3] * f[1] + g[2] * g[2] * f[3] + g[1] * g[1] * f[3] + g[0] * g[0] * f[3] + g[1] * g[2] * f[0] + g[1] * g[2] * f[0];
        f[0] = f0;
        f[1] = f1;
        f[2] = f2;
        f[3] = f3;
    }
    gettimeofday(&end, 0);
    solveTime = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec) * 1e-6;

    std::cout << "f: " << f << std::endl;
    std::cout << "g: " << g << std::endl;
    std::cout << "handcoded: multiply solve time: " << solveTime << std::endl;

    gettimeofday(&start, 0);
    for (int i = 0; i < 1e8; ++i)
    {
        double f0 = -2.0 * g[1] * g[3] * f[2] + 2.0 * g[2] * g[3] * f[1] + g[3] * g[3] * f[3] - 2.0 * g[1] * g[2] * f[3] + g[3] * g[3] * f[0] + g[2] * g[2] * f[0] + g[1] * g[1] * f[0] + g[0] * g[0] * f[0];
        double f1 = 2.0 * g[1] * g[2] * f[2] + g[3] * g[3] * f[1] + g[2] * g[2] * f[1] + g[1] * g[1] * f[1] + g[0] * g[0] * f[1] - 2.0 * g[1] * g[3] * f[3] - 2.0 * g[2] * g[3] * f[0];
        double f2 = g[3] * g[3] * f[2] + g[1] * g[1] * f[2] + g[0] * g[0] * f[2] - 2.0 * g[1] * g[2] * f[1] - 2.0 * g[2] * g[3] * f[3] + 2.0 * g[1] * g[3] * f[0];
        double f3 = 2.0 * g[2] * g[3] * f[2] + g[2] * g[2] * f[2] + 2.0 * g[1] * g[3] * f[1] + g[2] * g[2] * f[3] + g[1] * g[1] * f[3] + g[0] * g[0] * f[3] + 2.0 * g[1] * g[2] * f[0];
        f[0] = f0;
        f[1] = f1;
        f[2] = f2;
        f[3] = f3;
    }
    gettimeofday(&end, 0);
    solveTime = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec) * 1e-6;

    std::cout << "f: " << f << std::endl;
    std::cout << "g: " << g << std::endl;
    std::cout << "handcoded optimized: multiply solve time: " << solveTime << std::endl;
}
