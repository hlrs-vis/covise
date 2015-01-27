/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "SMP.h"
#include "TetraNeighbor.h"
#include <appl/ApplInterface.h>

#if defined(__sgi)
#include <ulocks.h>
#endif

void TetraNeighbor::goSMP(const coDistributedObject *const *setIn, coDoFloat **setOut, int numStart, int numNodes)
{
#if defined(__sgi)
    char *arenaName = NULL;
    usptr_t *myArena;
    barrier_t *myBarrier;
    SMPInfo *info;

    int i, j;

    // get an arena
    arenaName = tempnam("", "LOCK");
    usconfig(CONF_INITUSERS, numNodes + 5);
    usconfig(CONF_ARENATYPE, US_SHAREDONLY);
    myArena = usinit(arenaName);

    // a barrier
    myBarrier = new_barrier(myArena);
    init_barrier(myBarrier);

    // load the parameters
    info = new SMPInfo[numNodes];
    j = numStart / numNodes;
    for (i = 0; i < numNodes; i++)
    {
        info[i].classPtr = this;
        info[i].b = myBarrier;

        info[i].setIn = setIn;
        info[i].setOut = setOut;
        info[i].numNodes = numNodes;

        if (i)
            info[i].from = info[i - 1].to;
        else
            info[i].from = 0;

        if (i == numNodes - 1)
            info[i].to = numStart;
        else
            info[i].to = (i + 1) * j;
    }

    // start processing
    for (i = 0; i < numNodes - 1; i++)
        sprocsp(SMPTraceNoClass, PR_SALL, (void *)&info[i], NULL, 1000000);

    // we do something, too
    SMPTraceNoClass((void *)&info[i]);

    // clean up
    free_barrier(myBarrier);
    if (myArena)
        unlink(arenaName);
    delete[] info;
#else
    (void)setIn;
    (void)setOut;
    (void)numStart;
    (void)numNodes;
#endif
    // done
    return;
}

void TetraNeighbor::runSMP(void *p)
{
#if defined(__sgi)
    int f, t;
    int i;

    coDoUnstructuredGrid *gridIn;
    float *newNeighborList;

    f = ((SMPInfo *)p)->from;
    t = ((SMPInfo *)p)->to;

    for (i = f; i < t; i++)
    {
        gridIn = (coDoUnstructuredGrid *)(((SMPInfo *)p)->setIn[i]);
        ((SMPInfo *)p)->setOut[i]->getAddress(&newNeighborList);
        localNeighbors(gridIn, newNeighborList, -1.0);

        fprintf(stderr, "TetraNeighbor::compute(SMP): localNeighbors(%d) done\n", i);
    }

    /*
   if( numGrids==1 )
      traceStationary( f, t );
   else
      traceTransient( f, t );
      //fprintf( stderr, "ugh - transient SMP not yet supported\n");
   */

    barrier(((SMPInfo *)p)->b, ((SMPInfo *)p)->numNodes);
#else
    (void)p;
#endif

    return;
}

void SMPTraceNoClass(void *p, size_t)
{
#if defined(__sgi)
    // call the class
    ((SMPInfo *)p)->classPtr->runSMP(p);
#else
    (void)p;
#endif

    // done
    return;
}
