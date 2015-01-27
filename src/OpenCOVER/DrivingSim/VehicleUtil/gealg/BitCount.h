/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __BitCount_h
#define __BitCount_h

template <unsigned int u>
struct BitCount
{
    static const unsigned int value = (((u - ((u >> 1) & 033333333333) - ((u >> 2) & 011111111111))
                                        + ((u - ((u >> 1) & 033333333333) - ((u >> 2) & 011111111111)) >> 3)) & 030707070707) % 63;
};

#endif
