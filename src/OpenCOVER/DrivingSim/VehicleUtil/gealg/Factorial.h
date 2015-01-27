/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __Factorial_h
#define __Factorial_h

template <unsigned int N>
struct Factorial
{
    static const unsigned int value = N * Factorial<N - 1>::value;
};

template <>
struct Factorial<0>
{
    static const unsigned int value = 1;
};

#endif
