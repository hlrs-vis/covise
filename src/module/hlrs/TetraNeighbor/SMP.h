/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#if !defined(__SMPTRACE_H)
#define __SMPTRACE_H

#include "TetraNeighbor.h"
#include <appl/ApplInterface.h>
using namespace covise;

#if defined(__sgi)
#include <ulocks.h>
#endif

struct SMPInfo
{
    TetraNeighbor *classPtr;
    coDistributedObject *const *setIn;
    coDoFloat **setOut;
    int from, to;
    int numNodes;

#if defined(__sgi)
    barrier_t *b;
#endif
};

void SMPTraceNoClass(void *p, size_t qwery = 0);
#endif // __SMPTRACE_H
