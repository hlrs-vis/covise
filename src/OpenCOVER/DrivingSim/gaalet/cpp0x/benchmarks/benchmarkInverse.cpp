/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "gaalet.h"
#include <sys/time.h>
#include <cmath>

int main()
{
    timeval start, end;
    double solveTime;

    gaalet::mv<1, 2, 4, 7>::type a = { 1.0, 1.0, 0.0, 0.0 };
    gaalet::mv<0, 3, 5, 6>::type b = { cos(-M_PI * 0.25), sin(-M_PI * 0.25), 0.0, 0.0 };

    gettimeofday(&start, 0);
    for (int i = 0; i < 1e8; ++i)
    {
        a = b * a * (!b);
    }
    gettimeofday(&end, 0);
    solveTime = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec) * 1e-6;

    std::cout << "operator!(): multiply solve time: " << solveTime << std::endl;

    gettimeofday(&start, 0);
    for (int i = 0; i < 1e8; ++i)
    {
        a = b * a * inverse(b);
    }
    gettimeofday(&end, 0);
    solveTime = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec) * 1e-6;

    std::cout << "inverse(): multiply solve time: " << solveTime << std::endl;

    gettimeofday(&start, 0);
    for (int i = 0; i < 1e8; ++i)
    {
        a = b * a * (!b);
    }
    gettimeofday(&end, 0);
    solveTime = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec) * 1e-6;

    std::cout << "operator!(): multiply solve time: " << solveTime << std::endl;

    gettimeofday(&start, 0);
    for (int i = 0; i < 1e8; ++i)
    {
        a = b * a * inverse(b);
    }
    gettimeofday(&end, 0);
    solveTime = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec) * 1e-6;

    std::cout << "inverse(): multiply solve time: " << solveTime << std::endl;
}
