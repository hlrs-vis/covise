/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "SMPTrace.h"
#include "trace.h"

#if defined(__sgi)

#include <ulocks.h>

#define PRECOMP_TN 0

void trace::traceSMP()
{
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
#if PRECOMP_TN
    int onS;
    onS = numStart;
    numStart = numGrids;
#endif
    j = numStart / numNodes;
    for (i = 0; i < numNodes; i++)
    {
        info[i].classPtr = this;
        info[i].b = myBarrier;

        if (i)
            info[i].traceFrom = info[i - 1].traceTo;
        else
            info[i].traceFrom = 0;

        if (i == numNodes - 1)
            info[i].traceTo = numStart;
        else
            info[i].traceTo = (i + 1) * j;
    }

#if PRECOMP_TN
    numStart = onS;
#endif
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

    // done
    return;
}

void trace::runSMP(void *p)
{
    int f, t;

    f = ((SMPInfo *)p)->traceFrom;
    t = ((SMPInfo *)p)->traceTo;

    if (numGrids == 1)
        traceStationary(f, t);
    else
        traceTransient(f, t);
    //fprintf( stderr, "ugh - transient SMP not yet supported\n");

    barrier(((SMPInfo *)p)->b, numNodes);

    return;
}

#else

void trace::traceSMP()
{
    // dummy
}

void trace::runSMP(void *)
{
    // dummy
}
#endif

void SMPTraceNoClass(void *p, size_t)
{
// call the class
#if defined(__sgi)
    ((SMPInfo *)p)->classPtr->runSMP(p);
#else
    (void)p;
#endif

    // done
    return;
}
