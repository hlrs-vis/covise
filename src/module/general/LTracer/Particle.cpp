/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Particle.h"

#include <util/coviseCompat.h>

// Parameters for the RK4 algorithm
//    if kappa is below _KAPPA_MIN, the stepwidth is doubled
//    if kappa exceeds _KAPPA_MAX, the stepwidth is halfed and the step redone
// good values are   0.05 and 0.20   but they might be optimized for better results
//    (faster/more accurate)
//
//  _DT_SMALLEST gives the smallest possible DT (dt/_DT_SMALLEST) to be used while integrating
//  _DT_TRIES gives max. number of adaptions per step
//  good values are 10.0 and 100

//#define _KAPPA_MIN  0.05
//#define _KAPPA_MAX  0.20
//#define _KAPPA_MIN  0.01
//#define _KAPPA_MAX  0.04
#define _KAPPA_MIN 0.015f
#define _KAPPA_MAX 0.080f
//#define _KAPPA_MIN  0.002
//#define _KAPPA_MAX  0.010

#define _DT_SMALLEST 10000.0f
#define _DT_TRIES 100

Particle::Particle(Grid *g, GridBlockInfo *gbC, GridBlockInfo *gbN, int nG, int nR)
{
    // get current values
    pos.sbcIdx = 0;
    pos.x = gbC->x;
    pos.y = gbC->y;
    pos.z = gbC->z;
    vel.u = gbC->u;
    vel.v = gbC->v;
    vel.w = gbC->w;
    rot.rx = gbC->rx;
    rot.ry = gbC->ry;
    rot.rz = gbC->rz;
    grad.g11 = gbC->g11;
    grad.g12 = gbC->g12;
    grad.g13 = gbC->g13;
    grad.g21 = gbC->g21;
    grad.g22 = gbC->g22;
    grad.g23 = gbC->g23;
    grad.g31 = gbC->g31;
    grad.g32 = gbC->g32;
    grad.g33 = gbC->g33;

    needGradient = nG;
    needRot = nR;

    grid = g;
    gbiCur = gbC;
    gbiNext = gbN;
    pathRotGrad = NULL;
    gbiCache = NULL;
    gbiNextCache = NULL;

    // the particle is just fine :)
    stoppedFlag = 0;
    loopDetectedFlag = 0;
    loopOffset = -1;

    // init paths
    maxSteps = 128;
    numSteps = 0;
    path = new pPstr[maxSteps];
    if (needGradient || needRot)
        pathRotGrad = new pGstr[maxSteps];

    // and store start-position/status
    addCurToPath();

    // get initial stepsize (increment)
    lastH = gbC->block->proposeIncrement(gbC);
    if (lastH <= 0.0)
        lastH = g->getStepDuration(gbiCur->stepNo);

    // NOTE: gbiNext is set by Grid::startParticle if we have a transient case

    // debug
    float p0[3], p1[3], p2[3], p3[3], px[3];

    p0[0] = -0.025f;
    p0[1] = 0.325f;
    p0[2] = 0.05f;

    p1[0] = -4.61483e-11f;
    p1[1] = 0.325f;
    p1[2] = 0.05f;

    p2[0] = -4.10207e-11f;
    p2[1] = 0.3f;
    p2[2] = 0.05f;

    p3[0] = -0.025f;
    p3[1] = 0.325f;
    p3[2] = 0.025f;

    px[0] = 0.00172409f;
    px[1] = 0.318032f;
    px[2] = 0.05f;

    //   cerr << "----------------------------=>" << endl;
    //   gbC->block->tetraTrace(p0,p1,p2,p3,px);
    //   cerr << "<=----------------------------" << endl;

    // done
    return;
}

Particle::~Particle()
{
    if (pathRotGrad)
        delete[] pathRotGrad;
    delete[] path;
    if (gbiCur)
        delete gbiCur;
    if (gbiNext)
        delete gbiNext;
    if (gbiCache)
        delete gbiCache;
    if (gbiNextCache)
        delete gbiNextCache;
    return;
}

int Particle::hasStopped()
{
    return (stoppedFlag);
}

int Particle::getNumSteps()
{
    return (numSteps);
}

void Particle::getWayPoint(int i, int *sbcIdx, float *x, float *y, float *z, float *u, float *v, float *w,
                           float *rx, float *ry, float *rz, float *g11, float *g12, float *g13,
                           float *g21, float *g22, float *g23, float *g31, float *g32, float *g33)
{
    *sbcIdx = path[i].sbcIdx;
    *x = path[i].x;
    *y = path[i].y;
    *z = path[i].z;
    *u = path[i].u;
    *v = path[i].v;
    *w = path[i].w;
    if (pathRotGrad && rx && ry && rz)
    {
        *rx = pathRotGrad[i].rx;
        *ry = pathRotGrad[i].ry;
        *rz = pathRotGrad[i].rz;
    }
    if (pathRotGrad && g11 && g12 && g13 && g21 && g22 && g23 && g31 && g32 && g33)
    {
        *g11 = pathRotGrad[i].g11;
        *g12 = pathRotGrad[i].g12;
        *g13 = pathRotGrad[i].g13;
        *g21 = pathRotGrad[i].g21;
        *g22 = pathRotGrad[i].g22;
        *g23 = pathRotGrad[i].g23;
        *g31 = pathRotGrad[i].g31;
        *g32 = pathRotGrad[i].g32;
        *g33 = pathRotGrad[i].g33;
    }
    return;
}

void Particle::addCurToPath()
{
    // hmm....depends on wether stopped particles should be left visible or not
    if (stoppedFlag)
        return;

    //cerr << "*+*  Particle::addCurToPath   *+*" << endl;
    //cerr << "  pos: " << pos.x << " " << pos.y << " " << pos.z << "    vel: " << vel.u << " " << vel.v << " " << vel.w << endl;

    if (numSteps == maxSteps)
    {
        // resize
        pPstr *newP;
        pGstr *newG;
        newP = new pPstr[maxSteps * 2];
        if (needGradient || needRot)
            newG = new pGstr[maxSteps * 2];
        else
            newG = NULL;

        // copy the data
        memcpy(newP, path, maxSteps * sizeof(pPstr));
        if (needGradient)
            memcpy(newG, pathRotGrad, maxSteps * sizeof(pGstr));

        // delete the "old" data
        delete[] path;
        if (needGradient || needRot)
            delete[] pathRotGrad;

        // finish it....
        path = newP;
        pathRotGrad = newG;
        maxSteps *= 2;
    }

    // add it
    path[numSteps].x = pos.x;
    path[numSteps].y = pos.y;
    path[numSteps].z = pos.z;
    path[numSteps].sbcIdx = pos.sbcIdx;
    path[numSteps].u = vel.u;
    path[numSteps].v = vel.v;
    path[numSteps].w = vel.w;
    if (pathRotGrad && (needGradient || needRot))
    {
        pathRotGrad[numSteps].rx = rot.rx;
        pathRotGrad[numSteps].ry = rot.ry;
        pathRotGrad[numSteps].rz = rot.rz;
        pathRotGrad[numSteps].g11 = grad.g11;
        pathRotGrad[numSteps].g12 = grad.g12;
        pathRotGrad[numSteps].g13 = grad.g13;
        pathRotGrad[numSteps].g21 = grad.g21;
        pathRotGrad[numSteps].g22 = grad.g22;
        pathRotGrad[numSteps].g23 = grad.g23;
        pathRotGrad[numSteps].g31 = grad.g31;
        pathRotGrad[numSteps].g32 = grad.g32;
        pathRotGrad[numSteps].g33 = grad.g33;
    }

    numSteps++;

    // done
    return;
}

///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

void Particle::loopDetection()
{
    // not yet implemented

    cerr << "### Particle::loopDetection  ---  not implemented, call ignored" << endl;

    return;
}

void Particle::trace(float dt, float eps)
{
    float myDT;
    float timePassed, ourStepSize, stepSize, lastStepSize;
    int numTries;

    float h; //, s;
    float dx, dy, dz;

    //float u1, v1, w1, dx2, dy2, dz2;
    //float ut, vt, wt;

    float err;
    int i, r, a;
    float n, d0, d1, d2;

    float dc;

    // velocities
    float k10, k11, k12, k20, k21, k22, k30, k31, k32;
    float k40, k41, k42, k50, k51, k52, k60, k61, k62;

    // delta (according to 16.2.6, difference between RK5CK and embedded RK4CK)
    //float alpha, beta, delta;
    float yerr0, yerr1, yerr2;

    // adaptive stepsize control
    float errmax;

    // give the weird formula a try
    float yscal0, yscal1, yscal2;

    // cash/karper parameters
    static float a2 = 0.2f, a3 = 0.3f, a4 = 0.6f, a5 = 1.0f, a6 = 0.875f;
    static float c1 = 37.0f / 378.0f, c3 = 250.0f / 621.0f, c4 = 125.0f / 594.0f, c6 = 512.0f / 1771.0f;
    static float dc5 = -277.0f / 14336.0f;
    static float b21 = 0.2f, b31 = 3.0f / 40.0f, b41 = 0.3f, b51 = -11.0f / 54.0f, b61 = 1631.0f / 55296.0f;
    static float b32 = 9.0f / 40.0f, b42 = -0.9f, b52 = 2.5f, b62 = 175.0f / 512.0f;
    static float b43 = 1.2f, b53 = -70.0f / 27.0f, b63 = 575.0f / 13824.0f, b54 = 35.0f / 27.0f, b64 = 44275.0f / 110592.0f, b65 = 253.0f / 4096.0f;

    float dc1 = c1 - 2825.0f / 27648.0f, dc3 = c3 - 18575.0f / 48384.0f, dc4 = c4 - 13525.0f / 55296.0f, dc6 = c6 - 0.25f;

    GridBlockInfo *gbiCur0, *gbiNext0;

    float px[3], pv[3];
    px[0] = gbiCur->x;
    px[1] = gbiCur->y;
    px[2] = gbiCur->z;

    pv[0] = px[0] + vel.u;
    pv[1] = px[1] + vel.v;
    pv[2] = px[2] + vel.w;

    gbiCur->block->debugTetra(px, px, px, pv);

    // do we need interpolation ?!
    i = (grid->getNumSteps() > 1);

    if (i && !gbiNext)
    {
        cerr << "Particle::trace ### ERROR ### inter-timestep-interpolation required but gbiNext is NULL...interpolation disabled" << endl;
        i = 0;
        gbiNext = NULL;
    }

    // validate dt (->myDT)
    myDT = grid->getStepDuration(gbiCur->stepNo);
    if (myDT == 0.0)
        myDT = dt;
    if (lastH == 0.0)
        lastH = myDT;

    // perform the integration
    timePassed = 0.0;
    numTries = 0;
    stepSize = lastH;

    // init gbi0 (for SBC/blockchanges)
    gbiCur0 = gbiCur->createCopy();
    if (gbiNext)
        gbiNext0 = gbiNext->createCopy();
    else
        gbiNext0 = NULL;

    // perform the integration
    while (timePassed < myDT && numTries < _DT_TRIES)
    {
        //      cerr << "-------------------------------------->>> begin rk5 step" << endl;

        // make sure to trace exactly myDT
        lastStepSize = stepSize;
        if (timePassed + stepSize > myDT)
            ourStepSize = myDT - timePassed;
        else
            ourStepSize = stepSize;
        numTries++;

        // the stepsize to be used in the step is  ourStepSize
        h = ourStepSize;

        // no attA condition occured so far
        a = 0;

        // embedded RK5, Cash and Karp parameters
        // see: Numerical Recipies in C, pp.760

        // k1 = h*f(0,start)
        k10 = h * vel.u;
        k11 = h * vel.v;
        k12 = h * vel.w;

        //      cerr << "---   k1: " << k10 << " " << k11 << " " << k12 << "      h: " << h << endl;

        //      cerr << "[k2]" << endl;

        // k2 = h*f(a2*h, start+b21*k1)
        r = this->getVelocity(timePassed + a2 * h, myDT, b21 * k10, b21 * k11, b21 * k12, &k20, &k21, &k22, &dc, 1, gbiCur0, gbiNext0);
        if (r == 2)
        {
            // attA condition meat
            k10 = k20 * h;
            k11 = k21 * h;
            k12 = k22 * h;
            vel.u = k20;
            vel.v = k21;
            vel.w = k22;
            a = 1;
            n = 1.0f / sqrt(k20 * k20 + k21 * k21 + k22 * k22);
            d0 = k10 * n;
            d1 = k11 * n;
            d2 = k12 * n;
        }
        else if (r == 0)
        {
            // particle stopped
            stoppedFlag = 1;
            delete gbiCur0;
            delete gbiNext0;
            return;
        }
        k20 *= h;
        k21 *= h;
        k22 *= h;

        //      cerr << "[k3]" << endl;

        // k3 = h*f(a3*h, start+b31*k1+b32*k2)
        r = this->getVelocity(timePassed + a3 * h, myDT, b31 * k10 + b32 * k20, b31 * k11 + b32 * k21, b31 * k12 + b32 * k22, &k30, &k31, &k32, &dc, 1, gbiCur0, gbiNext0);
        if (r == 2)
        {
            // attA condition meat
            k20 = k30 * h;
            k21 = k31 * h;
            k22 = k32 * h;
            a = 1;
            n = 1 / sqrt(k30 * k30 + k31 * k31 + k32 * k32);
            d0 = k20 * n;
            d1 = k21 * n;
            d2 = k22 * n;
        }
        else if (r == 0)
        {
            // particle stopped
            stoppedFlag = 1;
            delete gbiCur0;
            delete gbiNext0;
            return;
        }
        else if (a)
        {
            // attA continued from previous intermediate step
            n = sqrt(k30 * k30 + k31 * k31 + k32 * k32) / h;
            k30 = d0 * n;
            k31 = d1 * n;
            k32 = d2 * n;
        }
        k30 *= h;
        k31 *= h;
        k32 *= h;

        //      cerr << "[k4]" << endl;

        // k4 = h*f(a4*h, start+b41*k1+b42*k2+b43*k3)
        // r = added by uwe (was macht das sonst fuer einen Sinn?
        r = this->getVelocity(timePassed + a4 * h, myDT, b41 * k10 + b42 * k20 + b43 * k30, b41 * k11 + b42 * k21 + b43 * k31, b41 * k12 + b42 * k22 + b43 * k32, &k40, &k41, &k42, &dc, 1, gbiCur0, gbiNext0);
        if (r == 2)
        {
            // attA condition meat
            k30 = k40 * h;
            k31 = k41 * h;
            k32 = k42 * h;
            a = 1;
            n = 1 / sqrt(k40 * k40 + k41 * k41 + k42 * k42);
            d0 = k30 * n;
            d1 = k31 * n;
            d2 = k32 * n;
        }
        else if (r == 0)
        {
            // particle stopped
            stoppedFlag = 1;
            delete gbiCur0;
            delete gbiNext0;
            return;
        }
        else if (a)
        {
            // attA continued from previous intermediate step
            n = sqrt(k40 * k40 + k41 * k41 + k42 * k42) / h;
            k40 = d0 * n;
            k41 = d1 * n;
            k42 = d2 * n;
        }
        k40 *= h;
        k41 *= h;
        k42 *= h;

        //      cerr << "[k5]" << endl;

        // k5 = h*f(a5*h, start+b51*k1+b52*k2+b53*k3+b54*k4)
        // r = added by uwe (was macht das sonst fuer einen Sinn?
        r = this->getVelocity(timePassed + a5 * h, myDT, b51 * k10 + b52 * k20 + b53 * k30 + b54 * k40, b51 * k11 + b52 * k21 + b53 * k31 + b54 * k41, b51 * k12 + b52 * k22 + b53 * k32 + b54 * k42, &k50, &k51, &k52, &dc, 1, gbiCur0, gbiNext0);
        if (r == 2)
        {
            // attA condition meat
            k40 = k50 * h;
            k41 = k51 * h;
            k42 = k52 * h;
            a = 1;
            n = 1 / sqrt(k50 * k50 + k51 * k51 + k52 * k52);
            d0 = k40 * n;
            d1 = k41 * n;
            d2 = k42 * n;
        }
        else if (r == 0)
        {
            // particle stopped
            stoppedFlag = 1;
            delete gbiCur0;
            delete gbiNext0;
            return;
        }
        else if (a)
        {
            // attA continued from previous intermediate step
            n = sqrt(k50 * k50 + k51 * k51 + k52 * k52) / h;
            k50 = d0 * n;
            k51 = d1 * n;
            k52 = d2 * n;
        }
        k50 *= h;
        k51 *= h;
        k52 *= h;

        //      cerr << "[k6]" << endl;

        // k6 = h*f(a6*h, start+b61*k1+b62*k2+b63*k3+b64*k4+b65*k5)
        // r = added by uwe (was macht das sonst fuer einen Sinn?
        r = this->getVelocity(timePassed + a6 * h, myDT, b61 * k10 + b62 * k20 + b63 * k30 + b64 * k40 + b65 * k50, b61 * k11 + b62 * k21 + b63 * k31 + b64 * k41 + b65 * k51, b61 * k12 + b62 * k22 + b63 * k32 + b64 * k42 + b65 * k52, &k60, &k61, &k62, &dc, 1, gbiCur0, gbiNext0);
        if (r == 2)
        {
            // attA condition meat
            k50 = k60 * h;
            k51 = k61 * h;
            k52 = k62 * h;
            a = 1;
            n = 1 / sqrt(k60 * k60 + k61 * k61 + k62 * k62);
            d0 = k50 * n;
            d1 = k51 * n;
            d2 = k52 * n;
        }
        else if (r == 0)
        {
            // particle stopped
            stoppedFlag = 1;
            delete gbiCur0;
            delete gbiNext0;
            return;
        }
        else if (a)
        {
            // attA continued from previous intermediate step
            n = sqrt(k60 * k60 + k61 * k61 + k62 * k62) / h;
            k60 = d0 * n;
            k61 = d1 * n;
            k62 = d2 * n;
        }
        k60 *= h;
        k61 *= h;
        k62 *= h;

        //      cerr << "---   k3: " << k30 << " " << k31 << " " << k32 << endl;
        //      cerr << "---   k4: " << k40 << " " << k41 << " " << k42 << endl;
        //      cerr << "---   k6: " << k60 << " " << k61 << " " << k62 << endl;

        // compute the step
        dx = c1 * k10 + c3 * k30 + c4 * k40 + c6 * k60;
        dy = c1 * k11 + c3 * k31 + c4 * k41 + c6 * k61;
        dz = c1 * k12 + c3 * k32 + c4 * k42 + c6 * k62;

        // estimate error as difference between RK5 and RK4
        yerr0 = dc1 * k10 + dc3 * k30 + dc4 * k40 + dc5 * k50 + dc6 * k60;
        yerr1 = dc1 * k11 + dc3 * k31 + dc4 * k41 + dc5 * k51 + dc6 * k61;
        yerr2 = dc1 * k12 + dc3 * k32 + dc4 * k42 + dc5 * k52 + dc6 * k62;

        /*
      errmax = fabs(yerr0/(dx+1.0e-30));  // add a small value to prevent div by 0
      err = fabs(yerr1/(dy+1.0e-30));
      if( err>errmax )
         errmax = err;
      err = fabs(yerr2/(dz+1.0e-30));
      if( err>errmax )
         errmax = err;
      */

        /* hmm... */
        yscal0 = fabs(pos.x) + fabs(k10) + 1.0e-30f;
        yscal1 = fabs(pos.y) + fabs(k11) + 1.0e-30f;
        yscal2 = fabs(pos.z) + fabs(k12) + 1.0e-30f;

        errmax = fabs(yerr0 / yscal0);
        err = fabs(yerr1 / yscal1);
        if (err > errmax)
            errmax = err;
        err = fabs(yerr2 / yscal2);
        if (err > errmax)
            errmax = err;
        // errmax = delta1 / delta0   (delta0=desired accuracy=eps)

        //fprintf(stderr, "yerr: %f %f %f\n", yerr0, yerr1, yerr2);
        //fprintf(stderr, "yscal: %f %f %f\n", yscal0, yscal1, yscal2);
        //fprintf(stderr, "h: %f  errmax: %f\n", h, errmax);

        /*
      cerr << "yerr: " << yerr0 << " " << yerr1 << " " << yerr2 << endl;
      cerr << "yscal: " << yscal0 << " " << yscal1 << " " << yscal2 << endl;
      cerr << "h: " << h << " errmax: " << errmax << endl;
      */

        errmax /= eps;

        /*
      //yscal0 = fabs(pos.x)+fabs(k10)+1.0e-30;
      //yscal1 = fabs(pos.y)+fabs(k11)+1.0e-30;
      //yscal2 = fabs(pos.z)+fabs(k12)+1.0e-30;
      yscal0 = fabs(dx)+1.0e-30;
      yscal1 = fabs(dy)+1.0e-30;
      yscal2 = fabs(dz)+1.0e-30;

      errmax = fabs(yerr0/yscal0);
      err = fabs(yerr1/yscal1);
      if( err>errmax )
      errmax = err;
      err = fabs(yerr2/yscal2);
      if( err>errmax )
      errmax = err;
      // errmax = delta1 / delta0   (delta0=desired accuracy=eps)
      errmax /= eps;
      */

        //      cerr << "[done]" << endl;
        // added !a because we just ignore the error if we follow the wall.
        if (!a && (errmax > 1.0 && h > myDT / _DT_SMALLEST)) //&& timePassed<0.77 ) // debug
        {
            // decrease stepsize and redo the current step
            stepSize = h * 0.9f * (float)powf(errmax, -0.25f);

            //         cerr << "   h: " << h << "   stepSize: " << stepSize << "    timePassed: " << timePassed << endl;

            if (stepSize < myDT / _DT_SMALLEST)
                stepSize = myDT / _DT_SMALLEST;

            //         cerr << "Particle::trace ### gbi reset/redo step  (" << errmax << ")" << endl;
            if (gbiCache)
            {
                if (gbiCur->blockNo != gbiCur0->blockNo || gbiCur->sbcIdx != gbiCur0->sbcIdx)
                    *gbiCache = *gbiCur;
            }
            else
                gbiCache = gbiCur->createCopy();
            *gbiCur = *gbiCur0;

            if (gbiNext)
            {
                if (gbiNextCache)
                {
                    if (gbiNext->blockNo != gbiNext0->blockNo || gbiNext->sbcIdx != gbiNext0->sbcIdx)
                        *gbiNextCache = *gbiNext;
                }
                else
                    gbiNextCache = gbiNext->createCopy();
                *gbiNext = *gbiNext0;
            }
        }
        else
        {
            // increase stepsize and continue with next step
            if (h == stepSize && errmax < 0.75)
                stepSize = h * 0.9f * (float)powf(errmax, -0.20f);

            // store current step
            timePassed += ourStepSize;
            if (!grid->advanceGBI(gbiCur, gbiNext, dx, dy, dz, gbiCache, gbiNextCache))
            {

                //gbiCur->block->debugTetra(gbiCur->x, gbiCur->y, gbiCur->z, 0.002);

                //cerr << "  pos: " << gbiCur->x << " , " << gbiCur->y << " , " << gbiCur->z << endl;

                //gbiCur->block->debugTetra(pos.x, pos.y, pos.z, 0.003);

                //cerr << "agbi stopped" << endl;

                stoppedFlag = 1;
                delete gbiCur0;
                delete gbiNext0;
                return;
            }
            //gbiCur->block->debugTetra(gbiCur->x, gbiCur->y, gbiCur->z, 0.001);

            //         cerr << "d: " << dx << " " << dy << " " << dz << endl;
            //         cerr << "### step performed. timePassed: " << timePassed << "    h: " << h << "   ourStepSize: " << ourStepSize << endl;

            pos.x += dx;
            pos.y += dy;
            pos.z += dz;
            pos.sbcIdx = gbiCur->sbcIdx;
            vel.u = gbiCur->u;
            vel.v = gbiCur->v;
            vel.w = gbiCur->w;

            //         cerr << " " << vel.u << " " << vel.v << " " << vel.w << endl;

            px[0] = pos.x;
            px[1] = pos.y;
            px[2] = pos.z;

            pv[0] = px[0] + vel.u;
            pv[1] = px[1] + vel.v;
            pv[2] = px[2] + vel.w;

            gbiCur->block->debugTetra(px, px, px, pv);

            /*
         if( gbiCache )
         {
            delete gbiCache;
            gbiCache = NULL;
         }
         if( gbiNextCache )
         {
            delete gbiNextCache;
            gbiNextCache = NULL;
         }
         */

            if (gbiCache)
            {
                if (gbiCur->blockNo != gbiCur0->blockNo || gbiCur->sbcIdx != gbiCur0->sbcIdx)
                    *gbiCache = *gbiCur;
                else
                {
                    gbiCache->x = gbiCur->x;
                    gbiCache->y = gbiCur->y;
                    gbiCache->z = gbiCur->z;
                }
            }
            else
                gbiCache = gbiCur->createCopy();
            if (gbiNext)
            {
                if (gbiNextCache)
                {
                    if (gbiNext->blockNo != gbiNext0->blockNo || gbiNext->sbcIdx != gbiNext0->sbcIdx)
                        *gbiNextCache = *gbiNext;
                    else
                    {
                        gbiNextCache->x = gbiNext->x;
                        gbiNextCache->y = gbiNext->y;
                        gbiNextCache->z = gbiNext->z;
                    }
                }
                else
                    gbiNextCache = gbiNext->createCopy();
            }

            *gbiCur0 = *gbiCur;
            if (gbiNext)
                *gbiNext0 = *gbiNext;
        }

        //cerr << "gbiCache: block= " << gbiCache->blockNo << endl;

        // cerr << errmax << "   new h: " << h << endl;

        //delta = sqrt(yerr0*yerr0 + yerr1*yerr1 + yerr2*yerr2);
        /* andere moeglichkeit, evtl. schneller: einfach maximale Fehlerkomponente verwenden
      delta = yerr0;
      delta = fmaxf(delta, yerr1);
      delta = fmaxf(delta, yerr2);
      */

        //cerr << delta << endl;

        //cerr << "d: " << dx << "  " << dy << "  " << dz << endl;

        //      cerr << "--------------------------------------<<< end rk5 step" << endl;
    }

    delete gbiCur0;
    delete gbiNext0;

    if (gbiCache)
    {
        delete gbiCache;
        gbiCache = NULL;
    }
    if (gbiNextCache)
    {
        delete gbiNextCache;
        gbiNextCache = NULL;
    }

    // remember to keep last stepsize -> used for the next steps
    //lastH = lastStepSize;
    lastH = stepSize;

    //   cerr << "Particle::trace=> numTries: " << numTries << "    lastH: " << lastH << endl;

    // stop the particle here as we couldn't compute the integration
    if (numTries >= _DT_TRIES)
    {
        stoppedFlag = 1;
        return;
    }

    // update GBIs (if necessary)
    if (i)
    {
        *gbiCur = *gbiNext;
        delete gbiNext;
        if (!grid->getGBINextStep(gbiCur, gbiNext))
        {
            stoppedFlag = 1;
            return;
        }
    }

    // remember to call getRotation / getGradient if necessary
    if (needGradient && gbiCur->block)
    {
        grid->getGradient(gbiCur);
        grad.g11 = gbiCur->g11;
        grad.g12 = gbiCur->g12;
        grad.g13 = gbiCur->g13;
        grad.g21 = gbiCur->g21;
        grad.g22 = gbiCur->g22;
        grad.g23 = gbiCur->g23;
        grad.g31 = gbiCur->g31;
        grad.g32 = gbiCur->g32;
        grad.g33 = gbiCur->g33;
    }
    if (needRot && gbiCur->block)
    {
        grid->getRotation(gbiCur);
        rot.rx = gbiCur->rx;
        rot.ry = gbiCur->ry;
        rot.rz = gbiCur->rz;
    }

    // load gbi-data into local data
    pos.sbcIdx = gbiCur->sbcIdx;
    pos.x = gbiCur->x;
    pos.y = gbiCur->y;
    pos.z = gbiCur->z;
    vel.u = gbiCur->u;
    vel.v = gbiCur->v;
    vel.w = gbiCur->w;

    //cerr << "sbc: " << pos.sbcIdx << "  pos: " << pos.x << " " << pos.y << " " << pos.z << endl;

    // store new position
    addCurToPath();

    // done
    return;
}

void Particle::oldTrace3(float dt)
{
    float myDT;
    float timePassed, ourStepSize, stepSize, lastStepSize;
    int numTries;

    float h, s = 0.0;
    float dx, dy, dz;

    float u1, v1, w1, dx2, dy2, dz2;
    float ut, vt, wt;

    float alpha, beta, err;
    int i, r;

    float dc;

    // do we need interpolation ?!
    i = (grid->getNumSteps() > 1);

    if (i && !gbiNext)
    {
        cerr << "Particle::trace ### ERROR ### inter-timestep-interpolation required but gbiNext is NULL...interpolation disabled" << endl;
        i = 0;
        gbiNext = NULL;
    }

    // validate dt (->myDT)
    myDT = grid->getStepDuration(gbiCur->stepNo);
    if (myDT == 0.0)
        myDT = dt;
    if (lastH == 0.0)
        lastH = myDT;
    //      lastH = (myDT/_DT_SMALLEST)*2.0;

    // perform the integration
    timePassed = 0.0;
    numTries = 0;
    stepSize = lastH;
    while (timePassed < myDT && numTries < _DT_TRIES)
    {
        // make sure to trace exactly myDT
        lastStepSize = stepSize;
        if (timePassed + stepSize > myDT)
            ourStepSize = myDT - timePassed;
        else
            ourStepSize = stepSize;
        numTries++;

        // rk4 2*(dt/2)
        // 1.
        h = ourStepSize / 2.0f;
        dx = 0.0;
        dy = 0.0;
        dz = 0.0;
        if (!traceRK4(vel.u, vel.v, vel.w, &dx, &dy, &dz, h, timePassed, myDT))
        {
            cerr << "1.rk4 stop" << endl;

            stoppedFlag = 1;
            return;
        }
        // remember velocity interpolation
        r = grid->getVelocity(gbiCur, dx, dy, dz, &u1, &v1, &w1, dc);
        if (r == 0)
        {
            stoppedFlag = 1;
            return;
        }
        if (i)
        {
            alpha = (timePassed + s) / myDT;
            beta = 1.0f - alpha;
            grid->getVelocity(gbiNext, dx, dy, dz, &ut, &vt, &wt, dc);
            u1 = beta * u1 + alpha * ut;
            v1 = beta * v1 + alpha * vt;
            w1 = beta * w1 + alpha * wt;
        }

        // 2.
        if (!traceRK4(u1, v1, w1, &dx, &dy, &dz, h, timePassed, myDT))
        {
            cerr << "2.rk4 stop" << endl;

            stoppedFlag = 1;
            return;
        }
        dx2 = dx;
        dy2 = dy;
        dz2 = dz;

        // rk4 (dt)
        dx = 0.0;
        dy = 0.0;
        dz = 0.0;
        if (!traceRK4(vel.u, vel.v, vel.w, &dx, &dy, &dz, ourStepSize, timePassed, myDT))
        {
            cerr << "rk4 stop" << endl;

            stoppedFlag = 1;
            return;
        }

        // compare results / adapt stepsize for next step
        alpha = sqrt((dx2 * dx2) + (dy2 * dy2) + (dz2 * dz2));
        beta = sqrt(((dx - dx2) * (dx - dx2)) + ((dy - dy2) * (dy - dy2)) + ((dz - dz2) * (dz - dz2)));
        //err = (beta*10000000.0/alpha);
        err = beta / alpha;
        //cerr << stepSize << "   " << err << endl;
        /*
      if( err>0.0001 )
         stepSize /= 2.0;
      else
      if( err<0.000005 && ourStepSize==stepSize )
         stepSize *= 2.0;
      */
        if (err > 0.001)
            stepSize /= 2.0;
        else if (err < 0.00005 && ourStepSize == stepSize)
            stepSize *= 2.0;

        // store current step
        timePassed += ourStepSize;
        if (!grid->advanceGBI(gbiCur, gbiNext, dx2, dy2, dz2))
        {

            gbiCur->block->debugTetra(gbiCur->x, gbiCur->y, gbiCur->z, 0.002f);

            cerr << "  pos: " << gbiCur->x << " , " << gbiCur->y << " , " << gbiCur->z << endl;

            gbiCur->block->debugTetra(pos.x, pos.y, pos.z, 0.003f);

            cerr << "agbi stopped" << endl;

            stoppedFlag = 1;
            return;
        }
        //   gbiCur->block->debugTetra(gbiCur->x, gbiCur->y, gbiCur->z, 0.001);

        pos.x += dx2;
        pos.y += dy2;
        pos.z += dz2;
        pos.sbcIdx = gbiCur->sbcIdx;
        vel.u = gbiCur->u;
        vel.v = gbiCur->v;
        vel.w = gbiCur->w;
    }

    //   cerr << "Particle::trace=> numTries: " << numTries << "    lastH: " << lastStepSize << endl;

    // remember to keep last stepsize -> used for the next steps
    //lastH = lastStepSize;
    lastH = stepSize;

    // update GBIs (if necessary)
    if (i)
    {
        *gbiCur = *gbiNext;
        delete gbiNext;
        if (!grid->getGBINextStep(gbiCur, gbiNext))
        {
            stoppedFlag = 1;
            return;
        }
    }

    // remember to call getRotation / getGradient if necessary
    if (needGradient && gbiCur->block)
    {
        grid->getGradient(gbiCur);
        grad.g11 = gbiCur->g11;
        grad.g12 = gbiCur->g12;
        grad.g13 = gbiCur->g13;
        grad.g21 = gbiCur->g21;
        grad.g22 = gbiCur->g22;
        grad.g23 = gbiCur->g23;
        grad.g31 = gbiCur->g31;
        grad.g32 = gbiCur->g32;
        grad.g33 = gbiCur->g33;
    }
    if (needRot && gbiCur->block)
    {
        grid->getRotation(gbiCur);
        rot.rx = gbiCur->rx;
        rot.ry = gbiCur->ry;
        rot.rz = gbiCur->rz;
    }

    // load gbi-data into local data
    pos.sbcIdx = gbiCur->sbcIdx;
    pos.x = gbiCur->x;
    pos.y = gbiCur->y;
    pos.z = gbiCur->z;
    vel.u = gbiCur->u;
    vel.v = gbiCur->v;
    vel.w = gbiCur->w;

    // store new position
    addCurToPath();

    // done
    return;
}

int Particle::getVelocity(float h, float dt, float dx, float dy, float dz,
                          float *u, float *v, float *w, float *dc, int sbc0, GridBlockInfo *gbiCur0, GridBlockInfo *gbiNext0)
{
    int r, s;
    float _u, _v, _w, alpha, beta;

    s = gbiCur->sbcIdx;
    r = grid->getVelocity(gbiCur, dx, dy, dz, u, v, w, *dc, sbc0, gbiCache, gbiCur0);
    if (!r)
        return (0);

    gbiCur->block->debugTetra(gbiCur->x + dx, gbiCur->y + dy, gbiCur->z + dz, 0.001f);
    float px[3], pv[3];
    px[0] = gbiCur->x + dx;
    px[1] = gbiCur->y + dy;
    px[2] = gbiCur->z + dz;

    pv[0] = px[0] + *u;
    pv[1] = px[1] + *v;
    pv[2] = px[2] + *w;

    gbiCur->block->debugTetra(px, px, px, pv);

    /*   if( gbiCur->sbcIdx!=s )
      {
          //cerr << "o   sbc wechsel festgestellt" << endl;
          if( gbiSBCCache )
             delete gbiSBCCache;
          gbiSBCCache = gbiCur->createCopy();
      }*/
    /*   else
      {
          if( gbiSBCCache )
          {
             delete gbiSBCCache;
             gbiSBCCache = NULL;
          }
      }
   */

    if (gbiNext)
    {
        alpha = h / dt;
        beta = 1.0f - alpha;

        //      cerr << "=   alpha: " << alpha << "   beta: " << beta << "     h: " << h << "   dt: " << dt << endl;

        if (grid->getVelocity(gbiNext, dx, dy, dz, &_u, &_v, &_w, *dc, sbc0, gbiNextCache, gbiNext0))
        {
            *u = beta * (*u) + alpha * _u;
            *v = beta * (*v) + alpha * _v;
            *w = beta * (*w) + alpha * _w;
        }
    }

    return (r);
}

int Particle::traceRK4(float u0, float v0, float w0, float *dx, float *dy, float *dz, float h, float stepTime, float stepDuration)
{
    float s, n;
    float alpha, beta;
    int i, r, a;

    float k10, k11, k12, k20, k21, k22, k30, k31, k32, k40, k41, k42;
    float kn0, kn1, kn2, d0, d1, d2;

    float dc;

    // do we need interpolation ?!
    i = (grid->getNumSteps() > 1);

    if (i && !gbiNext)
    {
        cerr << "Particle::trace ### ERROR ### inter-timestep-interpolation required but gbiNext is NULL...interpolation disabled" << endl;
        i = 0;
        gbiNext = NULL;
    }

    // a is a flag, it is set whenever the attackAngle condition is met, so that after
    //  it is met in one integration-step the further steps know that and adapt (to d[])
    a = 0;

    // RK4
    // h == stepSize
    // k1 = f(start)
    k10 = u0;
    k11 = v0;
    k12 = w0;

    // just to simplify....
    s = h / 2.0f;
    // k2 = f(start+h/2 * k1)
    r = grid->getVelocity(gbiCur, *dx + s * k10, *dy + s * k11, *dz + s * k12, &k20, &k21, &k22, dc);
    if (r == 2)
    {
        k10 = k20;
        k11 = k21;
        k12 = k22;
        a = 1;
        n = (float)(1.0 / sqrt(k20 * k20 + k21 * k21 + k22 * k22));
        d0 = k20 * n;
        d1 = k21 * n;
        d2 = k22 * n;
    }
    else if (r == 0)
        return (0);

    if (i)
    {
        alpha = (stepTime + s) / stepDuration;
        beta = 1.0f - alpha;
        grid->getVelocity(gbiNext, *dx + s * k10, *dy + s * k11, *dz + s * k12, &kn0, &kn1, &kn2, dc);
        k20 = beta * k20 + alpha * kn0;
        k21 = beta * k21 + alpha * kn1;
        k22 = beta * k22 + alpha * kn2;
    }

    // k3 = f(start+h/2 * k2)
    r = grid->getVelocity(gbiCur, *dx + s * k20, *dy + s * k21, *dz + s * k22, &k30, &k31, &k32, dc);
    if (r == 2)
    {
        k20 = k30;
        k21 = k31;
        k22 = k32;
        a = 1;
        n = 1 / sqrt(k30 * k30 + k31 * k31 + k32 * k32);
        d0 = k30 * n;
        d1 = k31 * n;
        d2 = k32 * n;
    }
    else if (r == 0)
        return (0);
    else if (a)
    {
        n = sqrt(k30 * k30 + k31 * k31 + k32 * k32);
        k30 = d0 * n;
        k31 = d1 * n;
        k32 = d2 * n;
    }
    if (i)
    {
        alpha = (stepTime + s) / stepDuration;
        beta = 1.0f - alpha;
        grid->getVelocity(gbiNext, *dx + s * k20, *dy + s * k21, *dz + s * k22, &kn0, &kn1, &kn2, dc);
        k30 = beta * k30 + alpha * kn0;
        k31 = beta * k31 + alpha * kn1;
        k32 = beta * k32 + alpha * kn2;
    }
    // k4 = f(start+h * k3)
    r = grid->getVelocity(gbiCur, *dx + h * k30, *dy + h * k31, *dz + h * k32, &k40, &k41, &k42, dc);
    if (r == 2)
    {
        k30 = k40;
        k31 = k41;
        k32 = k42;
    }
    else if (r == 0)
        return (0);
    else if (a)
    {
        n = sqrt(k40 * k40 + k41 * k41 + k42 * k42);
        k40 = d0 * n;
        k41 = d1 * n;
        k42 = d2 * n;
    }
    if (i)
    {
        alpha = (stepTime + s) / stepDuration;
        beta = 1.0f - alpha;
        grid->getVelocity(gbiNext, *dx + h * k30, *dy + h * k31, *dz + h * k32, &kn0, &kn1, &kn2, dc);
        k40 = beta * k40 + alpha * kn0;
        k41 = beta * k41 + alpha * kn1;
        k42 = beta * k42 + alpha * kn2;
    }

    // perform step
    s = h / 6.0f;
    *dx += s * (k10 + 2.0f * (k20 + k30) + k40);
    *dy += s * (k11 + 2.0f * (k21 + k31) + k41);
    *dz += s * (k12 + 2.0f * (k22 + k32) + k42);

    // get velocity at final position (this will also help advanceGBI) // HACK: sollte in grid::advanceGBI gemacht werden
    //grid->getVelocity(gbiCur, *dx, *dy, *dz, &k40, &k41, &k42, dc);
    //if( i )
    //   grid->getVelocity(gbiNext, *dx, *dy, *dz, &kn0, &kn1, &kn2, dc);

    if (a)
        return (2);

    // done
    return (1);
}

void Particle::oldTrace(float dt)
{
    float myDT;
    float timePassed, ourStepSize, stepSize;
    int numTries, storeFlag;

    float kappa;
    float k1[3], k2[3], k3[3], k4[3];
    //float ka[3];
    float h, s, n;
    float dx, dy, dz, dc;

    float alpha, beta;
    float kn[3], d[3];
    int i, r, a;

    // TEST HACK TEST
    float k10, k11, k12, k20, k21, k22, k30, k31, k32, k40, k41, k42;
    float ka0, ka1, ka2;
    /*
   i = grid->getVelocity(gbiCur, 0.0, 0.0, 0.0, &vel.u, &vel.v, &vel.w);
   cerr << numSteps << "=>     u: " << vel.u << "  v: " << vel.v << "  w: " << vel.w << "    gV: " << i << endl;
   if( numSteps==27 )
      cerr << "      pos1:  " << gbiCur->x << "  " << gbiCur->y << "  " << gbiCur->z << endl;
   */

    // bei dem 1. zwischenschritt bei dem er rausfliegt....hmmm
    //  sollte die integration "unterbrochen" werden und nur der gittergrenzen-parallele anteil
    //  genommen werden...bloss wie....hmmm....

    // do we need interpolation ?!
    i = (grid->getNumSteps() > 1);

    if (i && !gbiNext)
    {
        cerr << "Particle::trace ### ERROR ### inter-timestep-interpolation required but gbiNext is NULL...interpolation disabled" << endl;
        i = 0;
        gbiNext = NULL;
    }

    //   if( i )
    //      cerr << "TODO: add inter-timestep-interpolation to Particle::trace" << endl;

    // cerr << "position Cur: " << gbiCur->x << "  " << gbiCur->y << "  " << gbiCur->z << endl;
    // if( gbiNext )
    //    cerr << "         Nxt: " << gbiNext->x << "  " << gbiNext->y << "  " << gbiNext->z << endl;

    // validate dt (->myDT)
    myDT = grid->getStepDuration(gbiCur->stepNo);
    if (myDT == 0.0)
        myDT = dt;
    if (lastH == 0.0)
        lastH = (myDT / _DT_SMALLEST) * 2.0f;

    // perform the integration
    timePassed = 0.0;
    numTries = 0;
    stepSize = lastH;
    while (timePassed < myDT && numTries < _DT_TRIES)
    {
        // make sure to trace exactly myDT
        if (timePassed + stepSize > myDT)
            ourStepSize = myDT - timePassed;
        else
            ourStepSize = stepSize;
        numTries++;

        // a is a flag, it is set whenever the attackAngle condition is met, so that after
        //  it is met in one integration-step the further steps know that and adapt (to d[])
        a = 0;

        //            cerr << "T: " << numTries << "   time: " << timePassed << "    ourStepSize (h): " << ourStepSize << "   myDT: " << myDT << endl;

        // RK4
        // h = stepSize
        h = ourStepSize;
        // k1 = f(start)
        k1[0] = vel.u;
        k1[1] = vel.v;
        k1[2] = vel.w;

        //   cerr << "k1: " << vel.u << "  " << vel.v << "  " << vel.w << endl;

        // just to simplify....
        s = h / 2.0f;
        // k2 = f(start+h/2 * k1)
        r = grid->getVelocity(gbiCur, s * k1[0], s * k1[1], s * k1[2], k2, k2 + 1, k2 + 2, dc);
        if (r == 2)
        {
            //cerr << "rk4: attA in k2" << endl;

            k1[0] = k2[0];
            k1[1] = k2[1];
            k1[2] = k2[2];
            vel.u = k1[0];
            vel.v = k1[1];
            vel.w = k1[2];
            a = 1;
            n = 1.0f / sqrt(k2[0] * k2[0] + k2[1] * k2[1] + k2[2] * k2[2]);
            d[0] = k2[0] * n;
            d[1] = k2[1] * n;
            d[2] = k2[2] * n;
        }
        else if (r == 0)
        {
            stoppedFlag = 1;
            return;
        }

        //cerr << "k2: " << k2[0] << "  " << k2[1] << "  " << k2[2] << endl;

        if (i)
        {
            alpha = (timePassed + s) / myDT;
            beta = 1.0f - alpha;

            //cerr << "nxt->getVel: " << grid->getVelocity(gbiNext, s*k1[0], s*k1[1], s*k1[2], kn, kn+1, kn+2) << endl;

            // cerr << "kn: " << kn[0] << "  " << kn[1] << "  " << kn[2] << endl;
            grid->getVelocity(gbiNext, s * k1[0], s * k1[1], s * k1[2], kn, kn + 1, kn + 2, dc);
            k2[0] = beta * k2[0] + alpha * kn[0];
            k2[1] = beta * k2[1] + alpha * kn[1];
            k2[2] = beta * k2[2] + alpha * kn[2];
        }

        // k3 = f(start+h/2 * k2)
        r = grid->getVelocity(gbiCur, s * k2[0], s * k2[1], s * k2[2], k3, k3 + 1, k3 + 2, dc);
        if (r == 2)
        {
            //cerr << "rk4: attA in k3" << endl;

            k2[0] = k3[0];
            k2[1] = k3[1];
            k2[2] = k3[2];
            a = 1;
            n = 1 / sqrt(k3[0] * k3[0] + k3[1] * k3[1] + k3[2] * k3[2]);
            d[0] = k3[0] * n;
            d[1] = k3[1] * n;
            d[2] = k3[2] * n;
        }
        else if (r == 0)
        {
            stoppedFlag = 1;
            return;
        }
        else if (a)
        {
            //cerr << "rk4: attA continued in k3" << endl;

            n = sqrt(k3[0] * k3[0] + k3[1] * k3[1] + k3[2] * k3[2]);
            k3[0] = d[0] * n;
            k3[1] = d[1] * n;
            k3[2] = d[2] * n;
        }
        if (i)
        {
            alpha = (timePassed + s) / myDT;
            beta = 1.0f - alpha;
            grid->getVelocity(gbiNext, s * k2[0], s * k2[1], s * k2[2], kn, kn + 1, kn + 2, dc);
            k3[0] = beta * k3[0] + alpha * kn[0];
            k3[1] = beta * k3[1] + alpha * kn[1];
            k3[2] = beta * k3[2] + alpha * kn[2];
        }
        // k4 = f(start+h * k3)
        r = grid->getVelocity(gbiCur, h * k3[0], h * k3[1], h * k3[2], k4, k4 + 1, k4 + 2, dc);
        if (r == 2)
        {
            //cerr << "rk4: attA in k4" << endl;

            k3[0] = k4[0];
            k3[1] = k4[1];
            k3[2] = k4[2];
        }
        else if (r == 0)
        {
            stoppedFlag = 1;
            return;
        }
        else if (a)
        {
            //cerr << "rk4: attA continued in k3" << endl;

            n = sqrt(k4[0] * k4[0] + k4[1] * k4[1] + k4[2] * k4[2]);
            k4[0] = d[0] * n;
            k4[1] = d[1] * n;
            k4[2] = d[2] * n;
        }
        if (i)
        {
            alpha = (timePassed + s) / myDT;
            beta = 1.0f - alpha;
            grid->getVelocity(gbiNext, h * k3[0], h * k3[1], h * k3[2], kn, kn + 1, kn + 2, dc);
            k4[0] = beta * k4[0] + alpha * kn[0];
            k4[1] = beta * k4[1] + alpha * kn[1];
            k4[2] = beta * k4[2] + alpha * kn[2];
        }

        // --------------------------------------------------------------------
        // performance ?
        k10 = k1[0];
        k11 = k1[1];
        k12 = k1[2];
        k20 = k2[0];
        k21 = k2[1];
        k22 = k2[2];
        k30 = k3[0];
        k31 = k3[1];
        k32 = k3[2];
        k40 = k4[0];
        k41 = k4[1];
        k42 = k4[2];

        // perform step
        s = h / 6.0f;
        dx = s * (k10 + 2.0f * (k20 + k30) + k40);
        dy = s * (k11 + 2.0f * (k21 + k31) + k41);
        dz = s * (k12 + 2.0f * (k22 + k32) + k42);
        // tetratrace bestimmt hier jetzt noch StopU/V/W...braucht man das ? pruefen

        // compute kappa-s and minimum kappa
        if (k20 - k10 == 0.0)
            ka0 = _KAPPA_MAX * 2.0f;
        else
            ka0 = fabsf((2.0f * (k30 - k20)) / (k20 - k10));
        if (k21 - k11 == 0.0)
            ka1 = _KAPPA_MAX * 2.0f;
        else
            ka1 = fabsf((2.0f * (k31 - k21)) / (k21 - k11));
        if (k22 - k12 == 0.0)
            ka2 = _KAPPA_MAX * 2.0f;
        else
            ka2 = fabsf((2.0f * (k32 - k22)) / (k22 - k12));
        kappa = ka0;
        if (ka1 < kappa)
            kappa = ka1;
        if (ka2 < kappa)
            kappa = ka2;

        /* original
      // perform step
      s = h/6.0;
      dx = s*(k1[0] + 2.0*(k2[0]+k3[0]) + k4[0]);
      dy = s*(k1[1] + 2.0*(k2[1]+k3[1]) + k4[1]);
      dz = s*(k1[2] + 2.0*(k2[2]+k3[2]) + k4[2]);
      // tetratrace bestimmt hier jetzt noch StopU/V/W...braucht man das ? pruefen

      // compute kappa-s and minimum kappa
      if( k2[0]-k1[0] == 0.0 )
      ka[0] = _KAPPA_MAX*2.0;
      else
      ka[0] = fabsf( (2.0*(k3[0]-k2[0]))/(k2[0]-k1[0]) );
      if( k2[1]-k1[1] == 0.0 )
      ka[1] = _KAPPA_MAX*2.0;
      else
      ka[1] = fabsf( (2.0*(k3[1]-k2[1]))/(k2[1]-k1[1]) );
      if( k2[2]-k1[2] == 0.0 )
      ka[2] = _KAPPA_MAX*2.0;
      else
      ka[2] = fabsf( (2.0*(k3[2]-k2[2]))/(k2[2]-k1[2]) );
      kappa = ka[0];
      if( ka[1]<kappa ) kappa = ka[1];
      if( ka[2]<kappa ) kappa = ka[2];
      */
        //       cerr << "   ka: " << ka[0] << "  " << ka[1] << "  " << ka[2] << endl;
        //       cerr << "   kappa: " << kappa << endl;

        // cerr << "FEHLER! in Particle.cpp ist ein Problem bei rk4...irgendwie funktioniert die schrittweitenadaption nicht. Pruefen mit TGenDat !" << endl;

        // eigentlich sollten wir kappa_max nehmen ?! oder nicht...komisch
        //kappa = _KAPPA_MAX*2.0;

        // adapt stepsize
        storeFlag = 1;
        if (kappa < _KAPPA_MIN && ourStepSize == stepSize)
        {
            //cerr << "S  " << kappa << endl;
            stepSize *= 2.0;
        }
        else if (kappa > _KAPPA_MAX)
        {
            //if( stepSize>myDT/_DT_SMALLEST )
            if (stepSize > myDT / _DT_SMALLEST)
            {
                //stepSize = ourStepSize*0.5;
                stepSize *= 0.5;
                //cerr << "s  " << kappa << endl;
                storeFlag = 0;
            }
            else
                stepSize = myDT / _DT_SMALLEST;
            //stepSize = ourStepSize/_DT_SMALLEST;
        }

        // was this a valid step
        if (storeFlag)
        {

            //         cerr << "storeFlag -> dPos = " << dx << " " << dy << " " << dz << endl;

            timePassed += ourStepSize;
            if (!grid->advanceGBI(gbiCur, gbiNext, dx, dy, dz))
            {
                stoppedFlag = 1;
                return;
            }

            // PROBLEM : wenn gbiNext nicht gefunden wird...wie soll es dann weitergehen ?!
            //    bzw. wie geht es bisher weiter ? SBC wird dabei wohl nicht richtig beachtet
            //    und deshalb ergeben sich SBC-spruenge in der Ausgabe

            /*   if( !grid->advanceGBI(gbiCur, gbiNext, dx, dy, dz) )
            {
               cerr << " ?!? " << endl;

               stoppedFlag = 1;
               return;
            }*/

            pos.x += dx;
            pos.y += dy;
            pos.z += dz;
            pos.sbcIdx = gbiCur->sbcIdx;

            //   cerr << "+  pos   :  x: " << pos.x << "  y: " << pos.y << "  z: " << pos.z << endl;
            //   cerr << "+  GBIcur:  x: " << gbiCur->x << "  y: " << gbiCur->y << "  z: " << gbiCur->z << endl;
            //   cerr << "+  GBInxt:  x: " << gbiNext->x << "  y: " << gbiNext->y << "  z: " << gbiNext->z << endl;

            // HACK, das sollte alles eleganter laufen bzw nicht neu die vel interpoliert
            // werden sondern direkt aus dem GBI entnommen !
            //grid->getVelocity(gbiCur, 0.0, 0.0, 0.0, &vel.u, &vel.v, &vel.w);
            vel.u = gbiCur->u;
            vel.v = gbiCur->v;
            vel.w = gbiCur->w;
        }
    }
    // remember to keep last stepsize -> used for the next steps
    lastH = stepSize;

    //cerr << "nT: " << numTries << endl;

    //cerr << "Particle::trace  ---   timePassed=" << timePassed << "    myDT=" << myDT << "     numTries=" << numTries << endl;

    // update GBIs (if necessary)
    if (i)
    {
        //cerr << "### Particle::trace ---  gbi-update not yet implemented, thus only non-timestep data supported" << endl;

        //cerr << "### Particle::trace --- GBIs not advancing through timesteps yet (?? grid->advanceGBI ??)" << endl;
        //      cerr << "gbiCur: " << gbiCur->stepNo << "   gbiNext: " << gbiNext->stepNo << endl;

        // ok, hier muss wohl der nomovebug begraben sein....testen
        // gbiNext wird wohl oben beim grid->advanceGBI nicht vorwaertsbewegt....komisch

        *gbiCur = *gbiNext;

        //   cerr << "*  GBIcur:  x: " << gbiCur->x << "  y: " << gbiCur->y << "  z: " << gbiCur->z << endl;

        delete gbiNext;
        if (!grid->getGBINextStep(gbiCur, gbiNext))
        {
            //cerr << "getGBINextStep returned 0 !!!" << endl;

            stoppedFlag = 1;
            return;
        }

        //      cerr << "(after grid->getGBINextStep:)   gbiCur: " << gbiCur->stepNo << "   gbiNext: " << gbiNext->stepNo << endl;

        //if( gbiCur->sbcIdx > gbiNext->sbcIdx )
        //         gbiNext->sbcIdx = gbiCur->sbcIdx;

        // why was this here ? it is a bug maybe ?! let's try it

        //      cerr << "Particle: gbiCur.sbcIdx: " << gbiCur->sbcIdx << "   gbiNext.sbcIdx: " << gbiNext->sbcIdx << endl;

        /*
      gbiCur = gbiNext;
      if( gbiCur->stepNo==grid->getNumSteps()-1 )
         i = 0;
      else
         i = gbiCur->stepNo+1;
      Particle *tmp = grid->
      */

        /*
      if( i )
      {
         delete gbiCur;
         delete gbiCache;
         gbiCur = gbiNext;
         gbiCache = gbiCacheNext;
         i = (gbiCur->stepNo+1)%grid->getNumSteps();
         gbiNext = grid->initGBI( i, pos.x, pos.y, pos.z, nextGBI );
         gbiCacheNext = NULL;
      }
      // IGNORED FOR NOW !!!
      */
    }

    // remember to call getRotation / getGradient if necessary
    if (needGradient && gbiCur->block)
    {
        grid->getGradient(gbiCur);
        grad.g11 = gbiCur->g11;
        grad.g12 = gbiCur->g12;
        grad.g13 = gbiCur->g13;
        grad.g21 = gbiCur->g21;
        grad.g22 = gbiCur->g22;
        grad.g23 = gbiCur->g23;
        grad.g31 = gbiCur->g31;
        grad.g32 = gbiCur->g32;
        grad.g33 = gbiCur->g33;
    }
    if (needRot && gbiCur->block)
    {
        grid->getRotation(gbiCur);
        rot.rx = gbiCur->rx;
        rot.ry = gbiCur->ry;
        rot.rz = gbiCur->rz;
    }

    //   cerr << "local Pos:  x: " << pos.x << "  y: " << pos.y << "  z: " << pos.z << endl;

    // load gbi-data into local data
    pos.sbcIdx = gbiCur->sbcIdx;
    pos.x = gbiCur->x;
    pos.y = gbiCur->y;
    pos.z = gbiCur->z;
    vel.u = gbiCur->u;
    vel.v = gbiCur->v;
    vel.w = gbiCur->w;

    //   cerr << "GBI Pos (should be identical to local pos):  x: " << pos.x << "  y: " << pos.y << "  z: " << pos.z << endl;

    //   cerr << "==---------------------------------------------------------------" << endl;

    // store new position
    addCurToPath();

    // done
    return;
}

void Particle::oldTrace2(float dt)
{
    float myDT;
    float timePassed, ourStepSize, stepSize, lastStepSize = 0.0;
    int numTries, storeFlag;

    float kappa;
    float h, s, n;
    float dx, dy, dz;

    float alpha, beta;
    int i, r, a;

    float k10, k11, k12, k20, k21, k22, k30, k31, k32, k40, k41, k42;
    float ka0, ka1, ka2, kn0, kn1, kn2, d0, d1, d2;

    float dc;

    // do we need interpolation ?!
    i = (grid->getNumSteps() > 1);

    if (i && !gbiNext)
    {
        cerr << "Particle::trace ### ERROR ### inter-timestep-interpolation required but gbiNext is NULL...interpolation disabled" << endl;
        i = 0;
        gbiNext = NULL;
    }

    // validate dt (->myDT)
    myDT = grid->getStepDuration(gbiCur->stepNo);
    if (myDT == 0.0)
        myDT = dt;
    if (lastH == 0.0)
        lastH = myDT;
    //      lastH = (myDT/_DT_SMALLEST)*2.0;

    // perform the integration
    timePassed = 0.0;
    numTries = 0;
    stepSize = lastH;
    while (timePassed < myDT && numTries < _DT_TRIES)
    {
        // make sure to trace exactly myDT
        lastStepSize = stepSize;
        if (timePassed + stepSize > myDT)
            ourStepSize = myDT - timePassed;
        else
            ourStepSize = stepSize;
        numTries++;

        // a is a flag, it is set whenever the attackAngle condition is met, so that after
        //  it is met in one integration-step the further steps know that and adapt (to d[])
        a = 0;

        // RK4
        // h = stepSize
        h = ourStepSize;
        // k1 = f(start)
        k10 = vel.u;
        k11 = vel.v;
        k12 = vel.w;

        // just to simplify....
        s = h / 2.0f;
        // k2 = f(start+h/2 * k1)
        r = grid->getVelocity(gbiCur, s * k10, s * k11, s * k12, &k20, &k21, &k22, dc);
        if (r == 2)
        {
            k10 = k20;
            k11 = k21;
            k12 = k22;
            vel.u = k10;
            vel.v = k11;
            vel.w = k12;
            a = 1;
            n = (float)(1.0 / sqrt(k20 * k20 + k21 * k21 + k22 * k22));
            d0 = k20 * n;
            d1 = k21 * n;
            d2 = k22 * n;
        }
        else if (r == 0)
        {
            stoppedFlag = 1;
            return;
        }

        if (i)
        {
            alpha = (timePassed + s) / myDT;
            beta = 1.0f - alpha;
            grid->getVelocity(gbiNext, s * k10, s * k11, s * k12, &kn0, &kn1, &kn2, dc);
            k20 = beta * k20 + alpha * kn0;
            k21 = beta * k21 + alpha * kn1;
            k22 = beta * k22 + alpha * kn2;
        }

        // k3 = f(start+h/2 * k2)
        r = grid->getVelocity(gbiCur, s * k20, s * k21, s * k22, &k30, &k31, &k32, dc);
        if (r == 2)
        {
            k20 = k30;
            k21 = k31;
            k22 = k32;
            a = 1;
            n = 1 / sqrt(k30 * k30 + k31 * k31 + k32 * k32);
            d0 = k30 * n;
            d1 = k31 * n;
            d2 = k32 * n;
        }
        else if (r == 0)
        {
            stoppedFlag = 1;
            return;
        }
        else if (a)
        {
            n = sqrt(k30 * k30 + k31 * k31 + k32 * k32);
            k30 = d0 * n;
            k31 = d1 * n;
            k32 = d2 * n;
        }
        if (i)
        {
            alpha = (timePassed + s) / myDT;
            beta = 1.0f - alpha;
            grid->getVelocity(gbiNext, s * k20, s * k21, s * k22, &kn0, &kn1, &kn2, dc);
            k30 = beta * k30 + alpha * kn0;
            k31 = beta * k31 + alpha * kn1;
            k32 = beta * k32 + alpha * kn2;
        }
        // k4 = f(start+h * k3)
        r = grid->getVelocity(gbiCur, h * k30, h * k31, h * k32, &k40, &k41, &k42, dc);
        if (r == 2)
        {
            k30 = k40;
            k31 = k41;
            k32 = k42;
        }
        else if (r == 0)
        {
            stoppedFlag = 1;
            return;
        }
        else if (a)
        {
            n = sqrt(k40 * k40 + k41 * k41 + k42 * k42);
            k40 = d0 * n;
            k41 = d1 * n;
            k42 = d2 * n;
        }
        if (i)
        {
            alpha = (timePassed + s) / myDT;
            beta = 1.0f - alpha;
            grid->getVelocity(gbiNext, h * k30, h * k31, h * k32, &kn0, &kn1, &kn2, dc);
            k40 = beta * k40 + alpha * kn0;
            k41 = beta * k41 + alpha * kn1;
            k42 = beta * k42 + alpha * kn2;
        }

        // perform step
        s = h / 6.0f;
        dx = s * (k10 + 2.0f * (k20 + k30) + k40);
        dy = s * (k11 + 2.0f * (k21 + k31) + k41);
        dz = s * (k12 + 2.0f * (k22 + k32) + k42);

        // compute kappa-s and minimum kappa
        /*
      if( k20-k10 == 0.0 )
         ka0 = _KAPPA_MAX*2.0;
      else
         ka0 = fabsf( (2.0*(k30-k20))/(k20-k10) );
      if( k21-k11 == 0.0 )
         ka1 = _KAPPA_MAX*2.0;
      else
         ka1 = fabsf( (2.0*(k31-k21))/(k21-k11) );
      if( k22-k12 == 0.0 )
         ka2 = _KAPPA_MAX*2.0;
      else
      ka2 = fabsf( (2.0*(k32-k22))/(k22-k12) );
      kappa = ka0;
      if( ka1<kappa ) kappa = ka1;
      if( ka2<kappa ) kappa = ka2;
      */

        // compute angle between k2 and k3
        // or sth like that...

        ka0 = h * (k30 - k20);
        ka1 = h * (k31 - k21);
        ka2 = h * (k32 - k22);

        kappa = ((ka0 * ka0) + (ka1 * ka1) + (ka2 * ka2)) / ((dx * dx) + (dy * dy) + (dz * dz));
        cerr << kappa * 1000.0 << endl;

        // adapt stepsize
        storeFlag = 1;
        /* brute force test ;)
      if( kappa<_KAPPA_MIN && ourStepSize==stepSize )
      {
         stepSize *= 2.0;
      }
      else
      if( kappa>_KAPPA_MAX )
      {
         if( stepSize>myDT/_DT_SMALLEST )
         {
            stepSize *= 0.5;
      storeFlag = 0;
      }
      else
      stepSize = myDT/_DT_SMALLEST;
      }
      */
        if (kappa < 0.00001 && ourStepSize == stepSize)
        {
            stepSize *= 2.0;
            //         cerr << "d" << endl;
        }
        else if (kappa > 0.0003)
        {
            stepSize *= 0.5;
            storeFlag = 0;
            //         cerr << "h" << endl;
        }

        // was this a valid step
        if (storeFlag)
        {
            timePassed += ourStepSize;
            if (!grid->advanceGBI(gbiCur, gbiNext, dx, dy, dz))
            {
                stoppedFlag = 1;
                return;
            }

            pos.x += dx;
            pos.y += dy;
            pos.z += dz;
            pos.sbcIdx = gbiCur->sbcIdx;
            vel.u = gbiCur->u;
            vel.v = gbiCur->v;
            vel.w = gbiCur->w;
        }
    }

    //   cerr << "Particle::trace=> numTries: " << numTries << "    lastH: " << lastStepSize << endl;

    // remember to keep last stepsize -> used for the next steps
    lastH = lastStepSize;

    // update GBIs (if necessary)
    if (i)
    {
        *gbiCur = *gbiNext;
        delete gbiNext;
        if (!grid->getGBINextStep(gbiCur, gbiNext))
        {
            stoppedFlag = 1;
            return;
        }
    }

    // remember to call getRotation / getGradient if necessary
    if (needGradient && gbiCur->block)
    {
        grid->getGradient(gbiCur);
        grad.g11 = gbiCur->g11;
        grad.g12 = gbiCur->g12;
        grad.g13 = gbiCur->g13;
        grad.g21 = gbiCur->g21;
        grad.g22 = gbiCur->g22;
        grad.g23 = gbiCur->g23;
        grad.g31 = gbiCur->g31;
        grad.g32 = gbiCur->g32;
        grad.g33 = gbiCur->g33;
    }
    if (needRot && gbiCur->block)
    {
        grid->getRotation(gbiCur);
        rot.rx = gbiCur->rx;
        rot.ry = gbiCur->ry;
        rot.rz = gbiCur->rz;
    }

    // load gbi-data into local data
    pos.sbcIdx = gbiCur->sbcIdx;
    pos.x = gbiCur->x;
    pos.y = gbiCur->y;
    pos.z = gbiCur->z;
    vel.u = gbiCur->u;
    vel.v = gbiCur->v;
    vel.w = gbiCur->w;

    // store new position
    addCurToPath();

    // done
    return;
}
