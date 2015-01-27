/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __Power_h
#define __Power_h

template <int V, unsigned int N>
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

#endif
