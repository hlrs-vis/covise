/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "gaalet.h"

typedef gaalet::algebra<gaalet::signature<1, 1, 1> > dm;

int main()
{
    dm::mv<1>::type e1 = { 1.0 };
    dm::mv<2>::type e2 = { 1.0 };
    dm::mv<4>::type e3 = { 1.0 };

    std::cout << "e1*e1: " << e1 *e1 << std::endl;
    std::cout << "e2*e2: " << e2 *e2 << std::endl;
    std::cout << "e3*e3: " << e3 *e3 << std::endl;

    std::cout << "(e1+e2+e3)*e3: " << (e1 + e2 + e3) * e3 << std::endl;
    std::cout << "(e1*e2+e2*e3+e3*e1)*e3: " << (e1 * e2 + e2 * e3 + e3 * e1) * e3 << std::endl;

    std::cout << "(e1+e2+e3)^e3: " << ((e1 + e2 + e3) ^ e3) << std::endl;
    std::cout << "(e1+e2+e3)&e3: " << ((e1 + e2 + e3) & e3) << std::endl;
}
