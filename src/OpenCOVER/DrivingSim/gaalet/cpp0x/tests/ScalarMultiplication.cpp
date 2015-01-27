/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "gaalet.h"

int main()
{
    typedef gaalet::algebra<gaalet::signature<3, 0> > em;

    em::mv<1, 2, 4>::type a = { 1, 2, 3 };

    double s = 5.0;

    std::cout << "a: " << a << ", s: " << s << ", a*s: " << a *s << ", -s*a: " << -s *a << std::endl;

    em::mv<1>::type e1 = { 1.0 };
    em::mv<2>::type e2 = { 1.0 };

    std::cout << "e1: " << e1 << ", e2: " << e2 << ", e1*e2: " << e1 *e2 << ", e1^e2: " << (e1 ^ e2) << ", e1&e2: " << (e1 & e2) << std::endl;

    auto e1dote2 = e1 & e2;
    std::cout << "a*(e1&e2): " << a *(e1 & e2) << ", a+(e1&e2): " << a + (e1 & e2) << std::endl;
}
