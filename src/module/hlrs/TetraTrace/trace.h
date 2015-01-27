/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#if !defined(__TRACE_H)
#define __TRACE_H

#include <appl/ApplInterface.h>
using namespace covise;
#include "TetraGrid.h"

class trace
{
private:
    // parameters (for a brief description see TetraTrace.h)
    float startpoint1[3], startpoint2[3];
    float startNormal[3], startDirection[3];
    int numStart;
    int whatOut;
    int startStyle;
    int traceStyle;
    int numSteps;
    float stepDuration;
    int numNodes;
    int multiProcMode;
    int startStep;
    int searchMode;

    // computed waypoints
    float **wayX, **wayY, **wayZ; // points passed
    float **wayU, **wayV, **wayW; // velocity at points passed
    int **wayC; // cells passed
    int *currentWaypoint; // point actually processing
    int numWaypoints; // number of points passed
    int *wayNum; // number of valid points

    // startpoints
    float *startX, *startY, *startZ;
    int *startC;
    int numStartX, numStartY, numStartZ;

    // actual position/status
    float *actX, *actY, *actZ;
    float *actU, *actV, *actW;
    int *actC;
    char *actS;
    float *actDt;

    // grids to trace on
    TetraGrid **traceGrid;
    int numGrids;

    // compute startpoints
    void computeStartPoints(TetraGrid *grid);

    // compute stepsize
    float computeStepSize(int el, TetraGrid *grid, float u, float v, float w);

    // perform runge-kutta 4th order integration
    char rk4(int el, float x, float y, float z, float u, float v, float w,
             float startTime, float stepSize, TetraGrid *fromGrid, TetraGrid *toGrid,
             int &stopCell, float &stopX, float &stopY, float &stopZ, float &stopU, float &stopV, float &stopW,
             float &kappa, int traceNum);

    // build the output (see output.h/.cpp)
    coDistributedObject **buildOutput(char **names);
    coDistributedObject *buildOutput_Point(char *name);
    coDistributedObject *buildOutput_PointData(char *name);
    coDistributedObject **buildOutput_Lines(char *lname, char *dname);
    coDistributedObject **buildOutput_Fader(char *pName, char *dName);
    void getOutput(int tr, int &n, float *x, float *y, float *z, float *u, float *v, float *w);

    // perform a stationary trace
    void traceStationary(int fromTrace = -1, int toTrace = -1);

    // perform a transient trace
    void traceTransient(int fromTrace = -1, int toTrace = -1);

    // functions for controlling the traces (see ...Trace.h/.cpp)
    void traceWorkstation();
    void traceSMP();
    void traceMMP();

public:
    trace();
    ~trace();

    // set parameters/input-objects
    void setInput(int numG, TetraGrid **g);
    void setParameters(float sp1[3], float sp2[3], float sn[3], float sd[3],
                       int nstart, int wo, int sstyle, int ts, int nstep,
                       float sdur, int nn, int mpm, int sstep, int smode);

    // here we go
    coDistributedObject **run(char **names);

    // store actual value(s)
    int storeActual(int no = -1);

    // continue a trace
    char continueTrace(int el, float x, float y, float z, float u, float v, float w,
                       float &stepSize, TetraGrid *fromGrid, TetraGrid *toGrid,
                       int &stopCell, float &stopX, float &stopY, float &stopZ, float &stopU, float &stopV, float &stopW, int traceNum);

    // functions for performing the traces (see ...Trace.h/.cpp)
    void runSMP(void *p);
};
#endif // __TRACE_H
