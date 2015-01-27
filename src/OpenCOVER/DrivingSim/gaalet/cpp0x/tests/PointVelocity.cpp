/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "gaalet.h"
#include <cmath>

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

    cm::mv<0x18>::type E = ep * em;

    cm::mv<0x1f>::type Ic = e1 * e2 * e3 * ep * em;
    cm::mv<0x07>::type Ie = e1 * e2 * e3;

    auto x = eval(1.0 * e1 + 2.0 * e2 + 3.0 * e3);
    auto p = eval(x + 0.5 * (x & x) * einf + e0);

    auto dx = eval(0.1 * e1);

    double h = 0.001;
    for (double t = 0.0; t < 10.0; t += h)
    {
        x = (p ^ E) * E;

        std::cout << "p: " << p << ", x: " << x << ", einf: " << (-1.0) * (p & e0) << ", e0: " << (-1.0) * (p & einf) << std::endl;

        auto dp = dx + (x & dx) * einf + e0;
        p = p + dp * h;
    }
}
