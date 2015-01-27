/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#if !defined(__TETRA_TRACE_H)
#define __TETRA_TRACE_H

#include <appl/ApplInterface.h>
using namespace covise;
#include <appl/CoviseAppModule.h>
#include "TetraGrid.h"

class TetraTrace : public CoviseAppModule
{
private:
    // parameters
    float startpoint1[3], startpoint2[3]; // two startpoints
    float startNormal[3], startDirection[3]; // normal and direction of starting-plane or whatever
    long numStart; // number of Traces to perform
    int whatOut; // what data should we compute ?
    int startStyle; // style in which the startpoints are "arranged"
    int traceStyle; // how should we output the trace (e.g. points, lines,...)
    long numSteps; // number of steps to compute (stationary)
    // for transient data/grid: number of cycles
    float stepDuration; // stationary: stepsize for output
    // transient: time between two timesteps
    long numNodes; // number of nodes to use for multiprocessing
    int multiProcMode; // multiprocessing mode (none, shared memory, mpi,...)
    long startStep; // initial timestep to start tracing in
    //   (transient case only)
    int searchMode; // perform quicksearch or overall-search ?!

    // the grid(s) we have to trace on
    TetraGrid **traceGrid;
    int numGrids;

    // get parameter-settings from covise
    void getParameters();

    // get input
    int getInput(const coDistributedObject **objIn);

public:
    // initialize covise
    TetraTrace(int argc, char *argv[]);

    // clean up
    ~TetraTrace();

    // called whenever the module is executed
    coDistributedObject **compute(const coDistributedObject **, char **);

    // tell covise that we're up and running
    void run()
    {
        Covise::main_loop();
    }
};
#endif // __TETRA_TRACE_H
