/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "gaalet.h"

typedef gaalet::algebra<gaalet::signature<3, 0> > em;

int main()
{
    em::mv<1, 2, 4>::type a;
    a[0] = 1;
    a[1] = 2;
    a[2] = 3;

    std::cout << "a: " << a << std::endl;
    std::cout << "!a: " << (~a) * (1.0 / (a * (~a)).element<0x00>()) << std::endl;
    std::cout << "!a: " << (!a) << std::endl;

    em::mv<0, 3, 5, 6>::type R;
    R[0] = 1;
    R[1] = 2;
    R[2] = 3;
    R[3] = 4;

    std::cout << "R: " << R << std::endl;
    std::cout << "!R: " << (~R) * (1.0 / (R * (~R)).element<0x00>()) << std::endl;
    std::cout << "!R: " << (!R) << std::endl;
}
