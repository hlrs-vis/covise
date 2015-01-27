/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <appl/ApplInterface.h>
#include "trace.h"
#include "util.h"
#include <util/coviseCompat.h>

#define PRECOMP_TN 0

trace::trace()
{
}

trace::~trace()
{
}

void trace::computeStartPoints(TetraGrid *grid)
{
    int i, j;
    int n0, n1;
    float v[3], o[3];
    float v0[3], v1[3];

    float myDir[3];

    float s, r;
    float s0, s1;

    int sc, lsc; // current cell, last valid cell
    int lrsc; // last row cell
    int f;

    o[0] = startpoint1[0];
    o[1] = startpoint1[1];
    o[2] = startpoint1[2];

    v[0] = startpoint2[0] - o[0];
    v[1] = startpoint2[1] - o[1];
    v[2] = startpoint2[2] - o[2];

    s = 1.0f / (float)(numStart - 1);

    numStartX = numStartY = numStartZ = 0;

    lrsc = lsc = sc = -1;

    // here we go
    switch (startStyle)
    {

    case 2: // plane
    {
        // given:  startpoint1, startpoint2, startDirection, startNormal
        //
        // we have to compute: s0, s1 first, then n0, n1
        //

        // normize all vectors first
        r = 1.0f / sqrt((startDirection[0] * startDirection[0]) + (startDirection[1] * startDirection[1]) + (startDirection[2] * startDirection[2]));
        startDirection[0] *= r;
        startDirection[1] *= r;
        startDirection[2] *= r;

        r = 1.0f / sqrt((startNormal[0] * startNormal[0]) + (startNormal[1] * startNormal[1]) + (startNormal[2] * startNormal[2]));
        startNormal[0] *= r;
        startNormal[1] *= r;
        startNormal[2] *= r;

        myDir[0] = startNormal[1] * startDirection[2] - startNormal[2] * startDirection[1];
        myDir[1] = startNormal[2] * startDirection[0] - startNormal[0] * startDirection[2];
        myDir[2] = startNormal[0] * startDirection[1] - startNormal[1] * startDirection[0];
        // that should be normized by definition, so we have nothing to do

        // the following algorithm is standard, though it is taken
        // from GraphicGems I, side 304
        //

        // v1 x v2 = v0  (for speed up)
        v0[0] = startDirection[1] * myDir[2] - startDirection[2] * myDir[1];
        v0[1] = startDirection[2] * myDir[0] - startDirection[0] * myDir[2];
        v0[2] = startDirection[0] * myDir[1] - startDirection[1] * myDir[0];

        // v1 = p2 - p1
        v1[0] = startpoint2[0] - startpoint1[0];
        v1[1] = startpoint2[1] - startpoint1[1];
        v1[2] = startpoint2[2] - startpoint1[2];

        // compute r = det(v1, myDir, v0)
        r = (v1[0] * myDir[1] * v0[2]) + (myDir[0] * v0[1] * v1[2]) + (v0[0] * v1[1] * myDir[2]) - (v0[0] * myDir[1] * v1[2]) - (v1[0] * v0[1] * myDir[2]) - (myDir[0] * v1[1] * v0[2]);

        // and divide by v0^2
        s0 = (v0[0] * v0[0]) + (v0[1] * v0[1]) + (v0[2] * v0[2]);
        if (s0 == 0.0)
        {
            fprintf(stderr, "s0 = 0.0 !!!\n");
            s0 = 0.01f;
        }

        r /= s0;

        // so upper left point is at
        s1 = -r; // ulp = startpoint1 + s1*myDir

        // and compute r = det(v1, startDirection, v0)
        r = (v1[0] * startDirection[1] * v0[2]) + (startDirection[0] * v0[1] * v1[2]) + (v0[0] * v1[1] * startDirection[2]) - (v0[0] * startDirection[1] * v1[2]) - (v1[0] * v0[1] * startDirection[2]) - (startDirection[0] * v1[1] * v0[2]);
        r /= s0;

        // so lower right point is at
        s0 = r; // lrp = startpoint1 + s0*startDirection

        // we flipped 'em somehow....damn
        r = s0;
        s0 = -s1;
        s1 = -r;

        // now compute "stepsize" and "stepcount"
        n1 = (int)sqrt(fabsf(((float)numStart) * (s1 / s0)));
        if (n1 == 0)
            n1 = 1;
        n0 = numStart / n1;

        s0 /= ((float)n0);
        s1 /= ((float)n1);

        v0[0] = startDirection[0];
        v0[1] = startDirection[1];
        v0[2] = startDirection[2];

        v1[0] = myDir[0];
        v1[1] = myDir[1];
        v1[2] = myDir[2];

        numStart = n1 * n0;
        numStartX = n0;
        numStartY = n1;

        lrsc = sc = lsc = -1;

        for (j = 0; j < n1; j++)
        {
            lsc = lrsc;
            f = 0;
            for (i = 0; i < n0; i++)
            {
                startX[i + (j * n0)] = o[0] + v0[0] * s0 * (float)i + v1[0] * s1 * (float)j;
                startY[i + (j * n0)] = o[1] + v0[1] * s0 * (float)i + v1[1] * s1 * (float)j;
                startZ[i + (j * n0)] = o[2] + v0[2] * s0 * (float)i + v1[2] * s1 * (float)j;
                sc = grid->findActCell(startX[i + (j * n0)], startY[i + (j * n0)], startZ[i + (j * n0)], r, r, r, lsc);
                startC[i + (j * n0)] = sc;
                if (sc != -1)
                {
                    if (!f)
                    {
                        lrsc = sc;
                        f = 1;
                    }
                    lsc = sc;
                }
                else if (searchMode == 2) // make sure this startpoint doesn't exist
                {
                    sc = grid->findActCell(startX[i + (j * n0)], startY[i + (j * n0)], startZ[i + (j * n0)], r, r, r, -1);
                    startC[i + (j * n0)] = sc;
                    if (sc != -1)
                    {
                        if (!f)
                        {
                            lrsc = sc;
                            f = 1;
                        }
                        lsc = sc;
                    }
                }
            }
        }
    }
    break;

    default: // connecting line between both startpoints
    {
        for (i = 0; i < numStart; i++)
        {
            startX[i] = o[0] + v[0] * s * (float)i;
            startY[i] = o[1] + v[1] * s * (float)i;
            startZ[i] = o[2] + v[2] * s * (float)i;
            sc = grid->findActCell(startX[i], startY[i], startZ[i], r, r, r, lsc);
            startC[i] = sc;
            if (sc != -1)
                lsc = sc;
        }
    }
    break;
    }

    // resort startpoints
    j = 0;
    for (i = 0; i < numStart; i++)
    {
        if (startC[i] != -1)
        {
            startX[j] = startX[i];
            startY[j] = startY[i];
            startZ[j] = startZ[i];
            startC[j] = startC[i];
            j++;
        }
    }
    numStart = j;

    // done
    return;
}

float trace::computeStepSize(int el, TetraGrid *grid, float u, float v, float w)
{
    float vol, vel;
    float ds, dt;

    // we use 3rd-root to compute border-length of a volume-identic cube
    vol = fabsf(grid->getVolume(el));
    ds = powf(vol, 1.0f / 3.0f);

    // and compute velocity
    vel = sqrt((u * u) + (v * v) + (w * w));

    // dt = ds/v
    if (vel == 0.0)
        dt = stepDuration;
    else
        dt = ds / vel;

    // done
    return (dt);
}

char trace::rk4(int el, float x, float y, float z, float u, float v, float w,
                float startTime, float stepSize, TetraGrid *fromGrid, TetraGrid *toGrid,
                int &stopCell, float &stopX, float &stopY, float &stopZ, float &stopU, float &stopV, float &stopW,
                float &kappa, int traceNum)
{

    char r = 1;
    int i;
    int n;

    float k1[3], k2[3], k3[3], k4[3];
    float ka[3];
    float h, s;
    float p[3];

    float alpha = 0.0, beta = 0.0;

    // use usual naming
    h = stepSize;

    // k1=f(start)
    k1[0] = u;
    k1[1] = v;
    k1[2] = w;

    // position for next pick
    s = h / 2.0f;
    p[0] = x + (s * k1[0]);
    p[1] = y + (s * k1[1]);
    p[2] = z + (s * k1[2]);

    // k2 = f(start+h/2 * k1)
    i = fromGrid->findActCell(p[0], p[1], p[2], k2[0], k2[1], k2[2], el);

    if (i == -1)
        return (0);

    n = fromGrid->getTimeNeighbor(el, toGrid, traceNum);
    if (toGrid && n != -1)
    {
        alpha = (startTime + s) / stepDuration;
        beta = 1.0f - alpha;

        i = toGrid->findActCell(p[0], p[1], p[2], ka[0], ka[1], ka[2], n);

        if (i != -1)
        {
            k2[0] = beta * k2[0] + alpha * ka[0];
            k2[1] = beta * k2[1] + alpha * ka[1];
            k2[2] = beta * k2[2] + alpha * ka[2];
        }
    }

    // position for next pick
    p[0] = x + (s * k2[0]);
    p[1] = y + (s * k2[1]);
    p[2] = z + (s * k2[2]);

    // k3 = f(start+h/2 * k2)
    i = fromGrid->findActCell(p[0], p[1], p[2], k3[0], k3[1], k3[2], el);

    if (i == -1)
        return (0);

    if (toGrid && n != -1)
    {
        i = toGrid->findActCell(p[0], p[1], p[2], ka[0], ka[1], ka[2], n);

        if (i != -1)
        {
            k2[0] = beta * k2[0] + alpha * ka[0];
            k2[1] = beta * k2[1] + alpha * ka[1];
            k2[2] = beta * k2[2] + alpha * ka[2];
        }
    }

    // position for last pick
    p[0] = x + (h * k3[0]);
    p[1] = y + (h * k3[1]);
    p[2] = z + (h * k3[2]);

    // k4 = f(start+h * k3)
    i = fromGrid->findActCell(p[0], p[1], p[2], k4[0], k4[1], k4[2], el);

    if (i == -1)
        return (0);

    if (toGrid && n != -1)
    {
        alpha = (startTime + h) / stepDuration;
        beta = 1.0f - alpha;

        i = toGrid->findActCell(p[0], p[1], p[2], ka[0], ka[1], ka[2], n);

        if (i != -1)
        {
            k2[0] = beta * k2[0] + alpha * ka[0];
            k2[1] = beta * k2[1] + alpha * ka[1];
            k2[2] = beta * k2[2] + alpha * ka[2];
        }
    }

    // finally compute target point
    s = h / 6.0f;
    stopX = x + s * (k1[0] + 2.0f * (k2[0] + k3[0]) + k4[0]);
    stopY = y + s * (k1[1] + 2.0f * (k2[1] + k3[1]) + k4[1]);
    stopZ = z + s * (k1[2] + 2.0f * (k2[2] + k3[2]) + k4[2]);
    stopCell = fromGrid->findActCell(stopX, stopY, stopZ, stopU, stopV, stopW, el);

    if (stopCell == -1)
        r = 0;
    else if (toGrid && n != -1)
    {
        i = toGrid->findActCell(stopX, stopY, stopZ, ka[0], ka[1], ka[2], n);

        if (i != -1)
        {
            stopU = beta * stopU + alpha * ka[0];
            stopV = beta * stopV + alpha * ka[1];
            stopW = beta * stopW + alpha * ka[2];
        }
    }

    // compute kappa-s
    ka[0] = fabsf((2.0f * (k3[0] - k2[0])) / (k2[0] - k1[0]));
    ka[1] = fabsf((2.0f * (k3[1] - k2[1])) / (k2[1] - k1[1]));
    ka[2] = fabsf((2.0f * (k3[2] - k2[2])) / (k2[2] - k1[2]));

    // and find minimum kappa
    kappa = ka[0];
    if (ka[1] < kappa)
        kappa = ka[1];
    if (ka[2] < kappa)
        kappa = ka[2];

    // done
    return (r);
}

void trace::setInput(int numG, TetraGrid **g)
{
    numGrids = numG;
    traceGrid = g;
    return;
}

void trace::setParameters(float sp1[3], float sp2[3], float sn[3], float sd[3],
                          int nstart, int wo, int sstyle, int ts, int nstep,
                          float sdur, int nn, int mpm, int sstep, int smode)
{
    int i;

    for (i = 0; i < 3; i++)
    {
        startpoint1[i] = sp1[i];
        startpoint2[i] = sp2[i];
        startNormal[i] = sn[i];
        startDirection[i] = sd[i];
    }

    numStart = nstart;
    whatOut = wo;
    startStyle = sstyle;
    traceStyle = ts;
    numSteps = nstep;
    stepDuration = sdur;
    numNodes = nn;
    multiProcMode = mpm;
    startStep = sstep;
    searchMode = smode;

    // done
    return;
}

coDistributedObject **trace::run(char **names)
{
    coDistributedObject **returnObject = NULL;

    WristWatch *ww = NULL;

    int i;
    int noTrace = 0;

    // we want to keep the time
    ww = new WristWatch();

    // prepare for finding startpoints
    startX = new float[numStart];
    startY = new float[numStart];
    startZ = new float[numStart];
    startC = new int[numStart];

    // start time-keeping
    ww->start();

    // compute startpoints
    computeStartPoints(traceGrid[startStep]);

    // check for error
    if (!numStart)
    {
        Covise::sendInfo("no startpoints found");
        numStart = 1;
        noTrace = 1;
    }

    // and stop
    ww->stop("StartPoints");

    // compute how many waypoints we will get
    if (numGrids > 1)
        numWaypoints = numGrids * numSteps; // numSteps gives number of cycles
    else
        numWaypoints = numSteps;

    // allocate mem
    actX = new float[numStart];
    actY = new float[numStart];
    actZ = new float[numStart];
    actU = new float[numStart];
    actV = new float[numStart];
    actW = new float[numStart];
    actC = new int[numStart];
    actS = new char[numStart];
    actDt = new float[numStart];

    wayX = new float *[numStart];
    wayY = new float *[numStart];
    wayZ = new float *[numStart];
    wayU = new float *[numStart];
    wayV = new float *[numStart];
    wayW = new float *[numStart];
    wayC = new int *[numStart];
    wayNum = new int[numStart];
    currentWaypoint = new int[numStart];
    for (i = 0; i < numStart; i++)
    {
        wayX[i] = new float[numWaypoints];
        wayY[i] = new float[numWaypoints];
        wayZ[i] = new float[numWaypoints];
        wayU[i] = new float[numWaypoints];
        wayV[i] = new float[numWaypoints];
        wayW[i] = new float[numWaypoints];
        wayC[i] = new int[numWaypoints];
        wayNum[i] = 0;
        currentWaypoint[i] = 0;
    }

    // store startpoints as actual positions
    for (i = 0; i < numStart; i++)
    {
        actX[i] = startX[i];
        actY[i] = startY[i];
        actZ[i] = startZ[i];
        actC[i] = startC[i];
        actS[i] = (char)1;

        if (traceGrid[startStep]->isInCell(actC[i], actX[i], actY[i], actZ[i], actU[i], actV[i], actW[i]) == -1)
        {
            // error !?
            // this may be because we didn't find any startpoints
            if (!noTrace)
            {
                fprintf(stderr, "ERROR - check trace::run\n");
                actS[i] = (char)0;
            }
            else
                actS[i] = (char)1;
        }
        else
            actDt[i] = computeStepSize(actC[i], traceGrid[startStep], actU[i], actV[i], actW[i]);
    }

    // perform trace depending on selected mode
    if (noTrace)
    {
        storeActual();
        storeActual();
    }
    else
    {
        ww->start();
#if defined(__sgi)
        switch (multiProcMode)
        {
        case 2: // SMP
            traceSMP();
            break;
        case 3: // MMP
            traceMMP();
            break;
        default: // none or invalid
            traceWorkstation();
            break;
        }
#else
        traceWorkstation();
#endif

#if PRECOMP_TN
        TetraGrid *fromGrid, *toGrid;
        coDistributedObject **setElems;
        char bfr[1024];

        setElems = new coDistributedObject *[numGrids + 1];
        setElems[numGrids] = NULL;

        returnObject = new coDistributedObject *[2];
        returnObject[0] = NULL;

        for (i = 0; i < numGrids; i++)
        {
            fromGrid = traceGrid[i];
            if (i == numGrids - 1)
                toGrid = traceGrid[0];
            else
                toGrid = traceGrid[i + 1];

            sprintf(bfr, "%s_%d", names[1], i);
            setElems[i] = fromGrid->buildNewNeighborList(bfr, toGrid);
        }

        returnObject[1] = new coDoSet(names[1], setElems);

        for (i = 0; i < numGrids; i++)
            delete setElems[i];

#else
        ww->stop("all Traces");
#endif
    }

// build output
#if !PRECOMP_TN
    returnObject = buildOutput(names);
#endif

    // clean up
    delete ww;

    delete[] startX;
    delete[] startY;
    delete[] startZ;
    delete[] startC;

    delete[] actX;
    delete[] actY;
    delete[] actZ;
    delete[] actU;
    delete[] actV;
    delete[] actW;
    delete[] actC;
    delete[] actS;
    delete[] actDt;

    for (i = 0; i < numStart; i++)
    {
        delete[] wayX[i];
        delete[] wayY[i];
        delete[] wayZ[i];
        delete[] wayU[i];
        delete[] wayV[i];
        delete[] wayW[i];
        delete[] wayC[i];
    }
    delete[] wayX;
    delete[] wayY;
    delete[] wayZ;
    delete[] wayU;
    delete[] wayV;
    delete[] wayW;
    delete[] wayC;
    delete[] wayNum;
    delete[] currentWaypoint;

    // done
    return (returnObject);
}

int trace::storeActual(int no)
{
    int i, r;

    if (no == -1)
    {
        // store all waypoints
        for (i = 0; i < numStart; i++)
        {
            wayX[i][currentWaypoint[i]] = actX[i];
            wayY[i][currentWaypoint[i]] = actY[i];
            wayZ[i][currentWaypoint[i]] = actZ[i];
            wayU[i][currentWaypoint[i]] = actU[i];
            wayV[i][currentWaypoint[i]] = actV[i];
            wayW[i][currentWaypoint[i]] = actW[i];
            wayC[i][currentWaypoint[i]] = actC[i];

            if (actS[i] == 1)
                wayNum[i]++;

            currentWaypoint[i]++;
        }

        r = (currentWaypoint[0] != numWaypoints);
    }
    else
    {
        // just store one specific waypoint
        if (currentWaypoint[no] == numWaypoints)
            return (0);

        wayX[no][currentWaypoint[no]] = actX[no];
        wayY[no][currentWaypoint[no]] = actY[no];
        wayZ[no][currentWaypoint[no]] = actZ[no];
        wayU[no][currentWaypoint[no]] = actU[no];
        wayV[no][currentWaypoint[no]] = actV[no];
        wayW[no][currentWaypoint[no]] = actW[no];
        wayC[no][currentWaypoint[no]] = actC[no];

        if (actS[no] == 1)
            wayNum[no]++;

        currentWaypoint[no]++;

        r = (currentWaypoint[no] != numWaypoints);
    }

    // done
    return (r);
}

char trace::continueTrace(int el, float x, float y, float z, float u, float v, float w,
                          float &stepSize, TetraGrid *fromGrid, TetraGrid *toGrid,
                          int &stopCell, float &stopX, float &stopY, float &stopZ, float &stopU, float &stopV, float &stopW, int traceNum)
{
    float timePassed;
    float ourStepSize;
    float kappa;

    int numTries;

    char r;

    int storeVal;

    float xT, yT, zT, uT, vT, wT;
    int cT;

    stopCell = el;
    stopX = x;
    stopY = y;
    stopZ = z;
    stopU = u;
    stopV = v;
    stopW = w;

    r = 1;
    timePassed = 0.0;
    storeVal = 1;
    numTries = 0;

    while (timePassed < stepDuration && r == 1 && numTries < 100)
    {
        // don't go further then we should
        if (timePassed + stepSize > stepDuration)
            ourStepSize = stepDuration - timePassed;
        else
            ourStepSize = stepSize;

        // check for vel 0
        //fprintf(stderr, "%f\n", stopU+stopV+stopW);

        if (stopU + stopV + stopW == 0.0 && numGrids == 1)
            // vel=0 and only one grid -> this is the end
            r = 2;
        else
            r = rk4(stopCell, stopX, stopY, stopZ, stopU, stopV, stopW,
                    timePassed, ourStepSize, fromGrid, toGrid,
                    cT, xT, yT, zT, uT, vT, wT, kappa, traceNum);

        if (r == 1)
        {
            numTries++;

            // adapt stepsize
            storeVal = 1;
            if (kappa < 0.05 && ourStepSize == stepSize)
            {
                // double stepsize for next step
                stepSize *= 2.0;
            }
            else if (kappa > 0.2)
            {
                // halve stepsize and redo current step
                if (stepSize > stepDuration / 10.0f)
                {
                    stepSize = ourStepSize * 0.5f;
                    storeVal = 0;
                }
                else
                    stepSize = stepDuration / 10.0f;
            }

            // store values
            if (storeVal)
            {
                timePassed += ourStepSize;

                stopCell = cT;
                stopX = xT;
                stopY = yT;
                stopZ = zT;
                stopU = uT;
                stopV = vT;
                stopW = wT;
            }
        }
    }

    if (numTries == 100)
        fprintf(stderr, "trace::continueTrace: WARNING: the chosen stepsize leads to numeric problems, adjust it slightly to remove the problem\n");

    // done
    return (r);
}

void trace::traceStationary(int fromTrace, int toTrace)
{
    int i, r;

    int stopCell;
    float stopX, stopY, stopZ;
    float stopU, stopV, stopW;

    if (fromTrace == -1 || toTrace == -1)
    {
        while (storeActual())
        {
            for (i = 0; i < numStart; i++)
            {
                // if trace still running then continue
                if (actS[i])
                {
                    actS[i] = continueTrace(actC[i], actX[i], actY[i], actZ[i], actU[i], actV[i], actW[i],
                                            actDt[i], traceGrid[0], NULL,
                                            stopCell, stopX, stopY, stopZ, stopU, stopV, stopW, i);
                    actC[i] = stopCell;
                    actX[i] = stopX;
                    actY[i] = stopY;
                    actZ[i] = stopZ;
                    actU[i] = stopU;
                    actV[i] = stopV;
                    actW[i] = stopW;

                    // we may have vel=0, the trace stoped
                    if (actS[i] == 2)
                        actS[i] = 0;
                }
            }
        }
    }
    else
    {
        // trace just one part
        r = 0;

        for (i = fromTrace; i < toTrace; i++)
            r |= storeActual(i);

        while (r)
        {
            r = 0;
            for (i = fromTrace; i < toTrace; i++)
            {
                // if trace still running then continue
                if (actS[i])
                {
                    actS[i] = continueTrace(actC[i], actX[i], actY[i], actZ[i], actU[i], actV[i], actW[i],
                                            actDt[i], traceGrid[0], NULL,
                                            stopCell, stopX, stopY, stopZ, stopU, stopV, stopW, i);
                    actC[i] = stopCell;
                    actX[i] = stopX;
                    actY[i] = stopY;
                    actZ[i] = stopZ;
                    actU[i] = stopU;
                    actV[i] = stopV;
                    actW[i] = stopW;

                    // we may have vel=0, the trace stoped
                    if (actS[i] == 2)
                        actS[i] = 0;
                }
                r |= storeActual(i);
            }
        }
    }

    // done
    return;
}

void trace::traceTransient(int fromTrace, int toTrace)
{
    int i, l;
    int f, t, g;

    int stopCell;
    float stopX, stopY, stopZ;
    float stopU, stopV, stopW;

    if (fromTrace == -1 || toTrace == -1)
    {
        fromTrace = 0;
        toTrace = numStart;
    }

#if PRECOMP_TN
    for (i = fromTrace; i < toTrace; i++)
    {
        f = i;
        if (i == numGrids - 1)
            t = 0;
        else
            t = i + 1;

        traceGrid[f]->buildNewNeighborList(NULL, traceGrid[t]);
        fprintf(stderr, "%d\n", i);
    }

    return;
#endif

    fprintf(stderr, "greetings to thee, stranger...i'm proud to trace %d through %d for thou\n", fromTrace, toTrace);

    //for( i=fromTrace; i<toTrace; i++ )
    //   storeActual( i );
    fprintf(stderr, "NOTE: the initial startpoints are not displayed, there is a problem somewhere\n");
    fprintf(stderr, "      with the time-synchronisation but i don't know where");

    for (l = 0; l < numSteps; l++) // cycle through the grids
    {
        for (g = 0; g < numGrids; g++)
        {
            // set to- and from- grid
            f = g;
            if (g == numGrids - 1)
                t = 0;
            else
                t = g + 1;

            // continue all traces
            for (i = fromTrace; i < toTrace; i++)
            {
                // if trace still running or vel 0 then continue
                if (actS[i])
                {
                    actS[i] = continueTrace(actC[i], actX[i], actY[i], actZ[i], actU[i], actV[i], actW[i],
                                            actDt[i], traceGrid[f], traceGrid[t],
                                            stopCell, stopX, stopY, stopZ, stopU, stopV, stopW, i);
                    actC[i] = stopCell;
                    actX[i] = stopX;
                    actY[i] = stopY;
                    actZ[i] = stopZ;
                    actU[i] = stopU;
                    actV[i] = stopV;
                    actW[i] = stopW;

                    if (actS[i])
                        storeActual(i);
                    else
                    {
                        // the trace stopped here, use it for debugging
                        //actS[i] = 1;
                        //actU[i] = actV[i] = actW[i] = 1000.0;
                        //storeActual(i);
                        //actS[i] = 0;
                    }

                    // find cell where we are in the next grid
                    if (actS[i])
                    {
                        //fprintf(stderr, "cont %d\n", actS[i]);

                        actC[i] = traceGrid[t]->findActCell(actX[i], actY[i], actZ[i],
                                                            actU[i], actV[i], actW[i], traceGrid[f]->getTimeNeighbor(actC[i], traceGrid[t], i));
                        if (actC[i] == -1)
                            actS[i] = 0;

                        //fprintf(stderr, "%d\n", actS[i]);
                    }
                }
            }
        }
    }
}
