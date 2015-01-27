/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __CanonicalReorderingSign_h
#define __CanonicalReorderingSign_h

#include "BitCount.h"
#include "util/coTypes.h"

template <unsigned int sum, unsigned int b, unsigned int a>
struct CanonicalReorderingSum
{
    static const unsigned int value = CanonicalReorderingSum<(sum + BitCount<(a & b)>::value), b, (a >> 1)>::value;
};

template <unsigned int sum, unsigned int b>
struct CanonicalReorderingSum<sum, b, 0>
{
    static const unsigned int value = sum;
};

template <unsigned int a, unsigned int b>
struct CanonicalReorderingSign
{
    static const int value = ((CanonicalReorderingSum<0, b, (a >> 1)>::value & 1) == 0) ? 1 : -1;
};

template <uint8_t BL, uint8_t BH, uint8_t S>
struct MetricTensorSign
{
    static const uint8_t BI = BL & BH;

    static const int value = (BitCount<S & BI>::value % 2) ? -1 : 1;
};

#endif
