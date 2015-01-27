/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __GAALET_UTILITY_H
#define __GAALET_UTILITY_H

namespace gaalet
{

/// Routine (developed by MIT-members?) for fast counting of number of bits in a multivector element bitmap (equals grade of element)
template <conf_t u>
struct BitCount
{
    static const conf_t value = (((u - ((u >> 1) & 033333333333) - ((u >> 2) & 011111111111))
                                  + ((u - ((u >> 1) & 033333333333) - ((u >> 2) & 011111111111)) >> 3)) & 030707070707) % 63;
};

/// Template implementation of Dorst's canonical reordering for basis-nth-vector multiplication (represented by bitmaps)
template <conf_t sum, conf_t b, conf_t a>
struct CanonicalReorderingSum
{
    static const conf_t value = CanonicalReorderingSum<(sum + BitCount<(a & b)>::value), b, (a >> 1)>::value;
};

template <conf_t sum, conf_t b>
struct CanonicalReorderingSum<sum, b, 0>
{
    static const conf_t value = sum;
};

template <conf_t a, conf_t b>
struct CanonicalReorderingSign
{
    static const int value = ((CanonicalReorderingSum<0, b, (a >> 1)>::value & 1) == 0) ? 1 : -1;
};

//Power
template <int V, conf_t N>
struct Power
{
    static const int value = V * Power<V, N - 1>::value;
};

template <int V>
struct Power<V, 1>
{
    static const int value = V;
};

template <int V>
struct Power<V, 0>
{
    static const int value = 1;
};

} //end namespace gaalet

#endif
