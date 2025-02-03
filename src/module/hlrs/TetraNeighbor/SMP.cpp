/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "SMP.h"
#include "TetraNeighbor.h"
#include <appl/ApplInterface.h>

void TetraNeighbor::goSMP(const coDistributedObject *const *setIn, coDoFloat **setOut, int numStart, int numNodes)
{
    (void)setIn;
    (void)setOut;
    (void)numStart;
    (void)numNodes;
    // done
    return;
}

void TetraNeighbor::runSMP(void *p)
{
    (void)p;

    return;
}

void SMPTraceNoClass(void *p, size_t)
{
    (void)p;

    // done
    return;
}
