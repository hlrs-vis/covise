/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "gaalet.h"

int main()
{
    gaalet::mv<1, 2, 4>::type a = { 1, 2, 3 };

    //std::cout << "a: " << a << ", !a: " << !a << std::endl;
    std::cout << "a: " << a << std::endl;
    std::cout << "!a: " << (~a) * (1.0 / (a * (~a)).element<0x00>()) << std::endl;
    std::cout << "!a: " << (!a) << std::endl;

    gaalet::mv<0, 3, 5, 6>::type b = { 0, 3, 5, 6 };
    std::cout << "b: " << (b) << std::endl;
    std::cout << "!b: " << (!b) << std::endl;

    //inversion not implemented yet:
    //gaalet::mv<1, 2, 3, 4>::type c = {0, 1, 2, 3};
    //std::cout << "c: " << (c) << std::endl;
    //std::cout << "!c: " << (!c) << std::endl;
}
