/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "gaalet.h"

int main()
{
    gaalet::mv<3, 5>::type a = { 1.0, 2.0 };
    gaalet::mv<3, 4>::type b = { 4.0, 7.0 };

    auto c = a - b;

    std::cout << "a: " << a << std::endl;
    std::cout << "b: " << b << std::endl;
    std::cout << "c: " << c << std::endl;

    gaalet::mv<0>::type d;

    typedef gaalet::algebra<gaalet::signature<3, 0> > em;
    em::mv<1, 2, 4>::type e = { 1, 2, 3 };
    em::mv<1, 2, 4>::type f = { 3, 4, 5 };
    std::cout << "ef: " << e *f << std::endl;
}
