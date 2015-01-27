/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "gaalet.h"
#include "symbex.h"

int main()
{
    gaalet::symbex a("a");
    gaalet::symbex b("b");

    gaalet::symbex c = a + b;

    std::cout << "a: " << a << ", b: " << b << ", c: " << c << ", a+c: " << a + c << ", (a+c)*b: " << (a + c) * b << std::endl;

    typedef gaalet::algebra<gaalet::signature<0, 0>, gaalet::symbex> sr;

    sr::mv<0>::type sa = { "a" };
    sr::mv<0>::type sb = { "b" };

    auto sc = a + b;

    std::cout << "a: " << sa << ", b: " << sb << ", c: " << sc << ", a+c: " << sa + sc << ", (a+c)*b: " << (sa + sc) * sb << ", sa.element<1>(): " << sa.element<1>() << std::endl;

    typedef gaalet::algebra<gaalet::signature<3, 0>, gaalet::symbex> sem;
    sem::mv<1, 2, 4>::type x = { "x1", "x2", "x3" };
    sem::mv<1, 2, 4>::type t = { "t1", "t2", "t3" };
    sem::mv<0, 3, 5, 6>::type R = { "R0", "R12", "R13", "R23" };

    std::cout << "x: " << x << ", R: " << R << ", R*x*~R: " << R *x * ~R << std::endl;

    auto U = eval(0.5 * (t - grade<1>(R * x * ~R)) & (t - grade<1>(R * x * ~R)));
    std::cout << "U: " << U << std::endl;

    sem::mv<3, 5, 6>::type m = { "m12", "m13", "m23" };
    auto mag_m = magnitude(m);
    //std::cout << "m: " << m << ", mag_m: " << mag_m << std::endl;
}
