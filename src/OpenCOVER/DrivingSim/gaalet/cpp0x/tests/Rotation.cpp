/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "gaalet.h"

typedef gaalet::algebra<gaalet::signature<3, 0, 0> > em;

typedef em::mv<0x1, 0x2, 0x4>::type Vector;

static em::mv<0x1>::type e1 = { 1.0 };
static em::mv<0x2>::type e2 = { 1.0 };
static em::mv<0x4>::type e3 = { 1.0 };

int main()
{
    auto m = 0.5 * M_PI * (e1 ^ e2);
    auto R = eval(exp(-0.5 * m));

    Vector a = { 1.6, 3.2, 5.5 };
    auto b = eval(grade<1>(R * a * ~R));

    std::cout << "m: " << m << ", R: " << R << std::endl;
    std::cout << "a: " << a << ", b: " << b << std::endl;
}
