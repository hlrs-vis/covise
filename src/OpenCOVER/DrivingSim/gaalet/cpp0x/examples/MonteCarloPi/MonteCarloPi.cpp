/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "gaalet.h"
#include <iostream>
#include <cstdlib>

int main()
{
    typedef gaalet::algebra<gaalet::signature<4, 1> > cm;
    cm::mv<0x01>::type e1 = { 1.0 };
    cm::mv<0x02>::type e2 = { 1.0 };
    cm::mv<0x04>::type e3 = { 1.0 };
    cm::mv<0x08>::type ep = { 1.0 };
    cm::mv<0x10>::type em = { 1.0 };

    cm::mv<0x00>::type one = { 1.0 };

    cm::mv<0x08, 0x10>::type e0 = 0.5 * (em - ep);
    cm::mv<0x08, 0x10>::type einf = em + ep;

    srand(3000);

    double r = 1.0;
    auto S = eval(e0 - 0.5 * r * r * einf);

    int n_q = 1e7;
    int n_s = 0;

    for (int i = 0; i < n_q; ++i)
    {
        auto x = eval(((double)rand() / (double)RAND_MAX * e1 + (double)rand() / (double)RAND_MAX * e2 + (double)rand() / (double)RAND_MAX * e3) * r);
        auto P = x + 0.5 * (x & x) * einf + e0;
        double d = eval(S & P);
        if (d >= 0.0)
        {
            ++n_s;
        }
    }

    std::cout << "Pi: " << 6.0 * (double)n_s / (double)n_q << std::endl;
}
