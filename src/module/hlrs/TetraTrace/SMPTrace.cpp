/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "SMPTrace.h"
#include "trace.h"


void trace::traceSMP()
{
    // dummy
}

void trace::runSMP(void *)
{
    // dummy
}

void SMPTraceNoClass(void *p, size_t)
{
// call the class
    (void)p;

    // done
    return;
}
