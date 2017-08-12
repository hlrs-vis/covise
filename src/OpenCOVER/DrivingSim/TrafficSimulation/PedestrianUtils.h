/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef PedestrianUtils_h
#define PedestrianUtils_h

#include <math.h>

// Tolerance for floating point comparison
#define EPSILON 1.0e-8

class PedestrianUtils
{
public:
    static bool floatEq(const double a, const double b);
};

/**
 * Determine whether two floating point numbers are "close enough" to each other, within the tolerance given by EPSILON
 */
inline bool PedestrianUtils::floatEq(const double a, const double b)
{
    return fabs(a - b) < EPSILON;
}

#endif
