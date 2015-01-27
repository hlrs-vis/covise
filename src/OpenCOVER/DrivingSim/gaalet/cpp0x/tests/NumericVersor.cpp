/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "gaalet.h"
#include <cmath>

int main()
{
    gaalet::cm::mv<0x01>::type e1 = { 1.0 };
    gaalet::cm::mv<0x02>::type e2 = { 1.0 };
    gaalet::cm::mv<0x04>::type e3 = { 1.0 };
    gaalet::cm::mv<0x08>::type ep = { 1.0 };
    gaalet::cm::mv<0x10>::type em = { 1.0 };

    gaalet::cm::mv<0x00>::type one = { 1.0 };

    gaalet::cm::mv<0x08, 0x10>::type e0 = 0.5 * (em - ep);
    gaalet::cm::mv<0x08, 0x10>::type einf = em + ep;

    gaalet::cm::mv<0x18>::type E = ep * em;

    gaalet::cm::mv<0x1f>::type I = e1 * e2 * e3 * ep * em;
    gaalet::cm::mv<0x07>::type i = e1 * e2 * e3;

    auto R_pi = one * cos(-M_PI * 0.5) + e2 * e3 * sin(-M_PI * 0.5);

    auto T_b = one + 0.5 * einf * (0.3 * e3);

    auto T_u = one + 0.5 * einf * (0.5 * e3);

    auto T_l = one + 0.5 * einf * (0.4 * e3);
    gaalet::cm::mv<0x0, 0x6, 0xa, 0xc, 0x12, 0x14, 0x18, 0x1e>::type R_l = one * cos(-M_PI * 0.5 * 0.6) + e2 * e3 * sin(-M_PI * 0.5 * 0.5);

    for (int i = 0; i < 10; ++i)
    {
        std::cout << "i: " << i << ", R_l: " << R_l << std::endl;
        std::cout << "\tR_l*(~R_l): " << R_l *(~R_l) << std::endl;
        R_l = T_b * (!R_l) * R_pi * T_u * (~T_l);
    }
}
