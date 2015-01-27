/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "gaalet.h"

typedef gaalet::algebra<gaalet::signature<3, 0, 1> > ma;

int main()
{
    ma::mv<0>::type one = { 1.0 };
    ma::mv<1>::type e1 = { 1.0 };
    ma::mv<2>::type e2 = { 1.0 };
    ma::mv<4>::type e3 = { 1.0 };
    ma::mv<8>::type e0 = { 1.0 };
    ma::mv<0xf>::type I = (e1 ^ e2 ^ e3 ^ e0);

    auto x = 1.0 * e2 * e3 + 2.0 * e3 * e1 + 3.0 * e1 * e2;
    std::cout << "x: " << x << std::endl;
    auto X = one + I * x;
    std::cout << "X: " << X << std::endl;

    double phi = M_PI * 0.5;
    auto R = one * cos(-phi * 0.5) + sin(-phi * 0.5) * (0.0 * e2 * e3 + 0.0 * e3 * e1 + 1.0 * e1 * e2);
    std::cout << "R: " << R << std::endl;
    std::cout << "R*x*~R: " << R *x * ~R << std::endl;
    std::cout << "R*X*~R: " << R *X * ~R << std::endl;

    auto T = one + 0.5 * I * (10.0 * e2 * e3 + 0.0 * e3 * e1 + 0.0 * e1 * e2);
    std::cout << "T: " << T << std::endl;
    std::cout << "T*x*T: " << T *x *T << std::endl;
    std::cout << "T*X*T: " << T *X *T << std::endl;
    std::cout << "T*x*~T: " << T *x * ~T << std::endl;
    std::cout << "T*X*~T: " << T *X * ~T << std::endl;
    std::cout << "~T*x*T: " << ~T *x *T << std::endl;
    std::cout << "~T*X*T: " << ~T *X *T << std::endl;
    std::cout << "~T*x*~T: " << ~T *x * ~T << std::endl;
    std::cout << "~T*X*~T: " << ~T *X * ~T << std::endl;

    auto M = R * T;
    std::cout << "M: " << M << std::endl;
    std::cout << "M*x*~M: " << M *x * ~M << std::endl;
    std::cout << "M*X*~M: " << M *X * ~M << std::endl;
    std::cout << "R*T*x*T*~R: " << R *T *x *T * ~R << std::endl;
    std::cout << "R*T*X*T*~R: " << R *T *X *T * ~R << std::endl;
}
