/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "gaalet.h"

typedef gaalet::algebra<gaalet::signature<3, 0> > em;

typedef gaalet::algebra<gaalet::signature<0, 6, 2> > sa;

int main()
{
    em::mv<7>::type Ie = { 1.0 };
    em::mv<1, 2, 4>::type a = { 1.0, 2.0, 3.0 };

    std::cout << "a: " << a << ", *a: " << dual(a) << ", **a: " << dual(dual(a)) << std::endl;

    std::cout << "a*~I: " << a *(~Ie) << ", a*~I*I: " << a * (~Ie) * Ie << std::endl;
    std::cout << "a*~I: " << a *(~Ie) << ", a*~I*~I: " << a * (~Ie) * (~Ie) << std::endl;

    static const sa::mv<1>::type e1 = { 1.0 };
    static const sa::mv<2>::type e2 = { 1.0 };
    static const sa::mv<4>::type e3 = { 1.0 };
    static const sa::mv<0x40>::type e0 = { 1.0 };

    std::cout << "dual(e1*e2): " << dual(e1 * e2) << std::endl;
    std::cout << "dual(e3*e0): " << dual(e3 * e0) << std::endl;
    std::cout << "*(*(e1*e2) ^ *(e3*e0)): " << dual(dual(e1 * e2) ^ dual(e3 * e0)) << std::endl;
}
