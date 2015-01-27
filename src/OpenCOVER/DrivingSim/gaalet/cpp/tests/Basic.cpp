/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "gaalet.h"

typedef gaalet::algebra<gaalet::signature<0, 0> > em;

int main()
{
    em::mv<3, 5>::type a(1.0, 2.0);
    em::mv<3, 4>::type b(4.0, 7.0);

    em::mv<3, 4, 5>::type c = a - b;

    std::cout << "a: " << a << std::endl;
    std::cout << "b: " << b << std::endl;
    std::cout << "c: " << c << std::endl;

    em::mv<0>::type d;

    std::cout << "part<a_type>(c): " << part_type<em::mv<3, 5>::type>(c) << std::endl;
}
