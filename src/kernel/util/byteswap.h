/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __COVISE_BYTESWAP_INCLUDED
#define __COVISE_BYTESWAP_INCLUDED

#include "coTypes.h"

namespace
{

inline bool machineIsLittleEndian()
{
    int x = 1;
    return (1 == (int)*((char *)&x));
}

inline bool machineIsBigEndian()
{
    int x = 1;
    return (1 != (int)*((char *)&x));
}

inline void byteSwap(uint32_t &value)
{
    value = ((value & 0x000000ff) << 24) | ((value & 0x0000ff00) << 8) | ((value & 0x00ff0000) >> 8) | ((value & 0xff000000) >> 24);
}

inline void byteSwap(int32_t &value)
{
    int uval = (uint32_t)value;
    value = ((uval & 0x000000ff) << 24) | ((uval & 0x0000ff00) << 8) | ((uval & 0x00ff0000) >> 8) | ((uval & 0xff000000) >> 24);
}

inline void byteSwap(uint32_t *values, int no)
{
    for (int i = 0; i < no; i++, values++)
        *values = ((*values & 0x000000ff) << 24) | ((*values & 0x0000ff00) << 8) | ((*values & 0x00ff0000) >> 8) | ((*values & 0xff000000) >> 24);
}

inline void byteSwap(int32_t *values, int no)
{
    byteSwap((uint32_t *)values, no);
}

inline void byteSwap(uint64_t &value)
{
    value =
#if defined(_WIN32) && !defined(__MINGW32__)
        ((value & 0x00000000000000ff) << 56) | ((value & 0x000000000000ff00) << 40) | ((value & 0x0000000000ff0000) << 24) | ((value & 0x00000000ff000000) << 8) | ((value & 0x000000ff00000000) >> 8) | ((value & 0x0000ff0000000000) >> 24) | ((value & 0x00ff000000000000) >> 40) | ((value & 0xff00000000000000) >> 56);
#else
        ((value & 0x00000000000000ffll) << 56) | ((value & 0x000000000000ff00ll) << 40) | ((value & 0x0000000000ff0000ll) << 24) | ((value & 0x00000000ff000000ll) << 8) | ((value & 0x000000ff00000000ll) >> 8) | ((value & 0x0000ff0000000000ll) >> 24) | ((value & 0x00ff000000000000ll) >> 40) | ((value & 0xff00000000000000ll) >> 56);
#endif
}

inline void byteSwap(int64_t &value)
{
    value =
#if defined(_WIN32) && !defined(__MINGW32__)
        ((value & 0x00000000000000ff) << 56) | ((value & 0x000000000000ff00) << 40) | ((value & 0x0000000000ff0000) << 24) | ((value & 0x00000000ff000000) << 8) | ((value & 0x000000ff00000000) >> 8) | ((value & 0x0000ff0000000000) >> 24) | ((value & 0x00ff000000000000) >> 40) | ((value & 0xff00000000000000) >> 56);
#else
        ((value & 0x00000000000000ffll) << 56) | ((value & 0x000000000000ff00ll) << 40) | ((value & 0x0000000000ff0000ll) << 24) | ((value & 0x00000000ff000000ll) << 8) | ((value & 0x000000ff00000000ll) >> 8) | ((value & 0x0000ff0000000000ll) >> 24) | ((value & 0x00ff000000000000ll) >> 40) | ((value & 0xff00000000000000ll) >> 56);
#endif
}

inline void byteSwap(uint64_t *values, int no)
{
    int i;
    for (i = 0; i < no; i++, values++)
    {
        *values =
#if defined(_WIN32) && !defined(__MINGW32__)
            ((*values & 0x00000000000000ff) << 56) | ((*values & 0x000000000000ff00) << 40) | ((*values & 0x0000000000ff0000) << 24) | ((*values & 0x00000000ff000000) << 8) | ((*values & 0x000000ff00000000) >> 8) | ((*values & 0x0000ff0000000000) >> 24) | ((*values & 0x00ff000000000000) >> 40) | ((*values & 0xff00000000000000) >> 56);
#else
            ((*values & 0x00000000000000ffll) << 56) | ((*values & 0x000000000000ff00ll) << 40) | ((*values & 0x0000000000ff0000ll) << 24) | ((*values & 0x00000000ff000000ll) << 8) | ((*values & 0x000000ff00000000ll) >> 8) | ((*values & 0x0000ff0000000000ll) >> 24) | ((*values & 0x00ff000000000000ll) >> 40) | ((*values & 0xff00000000000000ll) >> 56);
#endif
    }
}

//We need this, since we want a _binary_
//conversion of the float value
#define double2ll(value) (*(uint64_t *)(&value))
inline void byteSwap(double &value)
{

    double2ll(value) =
#if defined(_WIN32) && !defined(__MINGW32__)
        ((double2ll(value) & 0x00000000000000ff) << 56) | ((double2ll(value) & 0x000000000000ff00) << 40) | ((double2ll(value) & 0x0000000000ff0000) << 24) | ((double2ll(value) & 0x00000000ff000000) << 8) | ((double2ll(value) & 0x000000ff00000000) >> 8) | ((double2ll(value) & 0x0000ff0000000000) >> 24) | ((double2ll(value) & 0x00ff000000000000) >> 40) | ((double2ll(value) & 0xff00000000000000) >> 56);
#else
        ((double2ll(value) & 0x00000000000000ffll) << 56) | ((double2ll(value) & 0x000000000000ff00ll) << 40) | ((double2ll(value) & 0x0000000000ff0000ll) << 24) | ((double2ll(value) & 0x00000000ff000000ll) << 8) | ((double2ll(value) & 0x000000ff00000000ll) >> 8) | ((double2ll(value) & 0x0000ff0000000000ll) >> 24) | ((double2ll(value) & 0x00ff000000000000ll) >> 40) | ((double2ll(value) & 0xff00000000000000ll) >> 56);
#endif
}

inline void byteSwap(double *values, int no)
{
    for (int i = 0; i < no; i++)
    {
        byteSwap(values[i]);
    }
}

#define float2int(value) (*(uint32_t *)(&value))
inline void byteSwap(float &value)
{

    float2int(value) = ((float2int(value) & 0x000000ff) << 24) | ((float2int(value) & 0x0000ff00) << 8) | ((float2int(value) & 0x00ff0000) >> 8) | ((float2int(value) & 0xff000000) >> 24);
}

inline void byteSwap(float *values, int no)
{
    uint32_t *intValues = (uint32_t *)values;
    for (int i = 0; i < no; i++, intValues++)
        *intValues = ((*intValues & 0x000000ff) << 24) | ((*intValues & 0x0000ff00) << 8) | ((*intValues & 0x00ff0000) >> 8) | ((*intValues & 0xff000000) >> 24);
}

inline void byteSwap(uint16_t &value)
{
    value = ((value & 0x00ff) << 8) | ((value & 0xff00) >> 8);
}

inline void byteSwap(uint16_t *values, int no)
{
    for (int i = 0; i < no; i++, values++)
        *values = ((*values & 0x00ff) << 8) | ((*values & 0xff00) >> 8);
}

inline void byteSwap(int16_t &value)
{
    uint16_t uval = (uint16_t)value;
    value = ((uval & 0x00ff) << 8) | ((uval & 0xff00) >> 8);
}

inline void byteSwap(int16_t *values, int no)
{
    byteSwap((uint16_t *)values, no);
}
}
#endif
