/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file coordconv.hxx
 * conversion between coordinate systems.
 * FAMU Copyright (C) 1998-2003 Institute for Theory of Electrical Engineering
 * @author K. Frenner
 * @author W. Hafla
 */

#include "coordconv.hxx" // conversion between coordinate systems.
#include <math.h>

#ifdef _WIN32
#pragma warning(disable : 981) // "operands are evaluated in unspecified order"
#pragma warning(disable : 869) // "parameter "rho" was never referenced"
#pragma warning(disable : 4100) // "unref. Parameter"
#endif

/**
 * conversion between coordinate systems.
 * @author K. Frenner
 */
namespace coordConv
{

void kart2zy(const double &x, const double &y, const double &z,
             double &rho2, double &phi2, double &z2)
{
    if (x == 0 && y == 0)
    {
        z2 = z;
        rho2 = 0;
        phi2 = 0;
        return;
    } //eigentlich undefiniert
    z2 = z;
    rho2 = sqrt(x * x + y * y);
    phi2 = atan2(y, x);
    if (phi2 < 0)
        phi2 = phi2 + 2 * 3.1415926535897932385;
}

void zy2kart(const double &rho, const double &phi, const double &z,
             double &x2, double &y2, double &z2)
{
    x2 = rho * cos(phi);
    y2 = rho * sin(phi);
    z2 = z;
}

} //namespace coordConv
