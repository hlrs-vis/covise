/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#if !defined(__SMPTRACE_H)
#define __SMPTRACE_H

#include "trace.h"

struct SMPInfo
{
    int dummy;
};

void SMPTraceNoClass(void *p, size_t qwery = 0);
#endif // __SMPTRACE_H
