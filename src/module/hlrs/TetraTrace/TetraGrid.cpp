/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <appl/ApplInterface.h>
#include "TetraGrid.h"
#include <util/coviseCompat.h>
#include <config/CoviseConfig.h>

#define PRECOMP_TN 0
//#define BOX_INCREASE_FACTOR  100

TetraGrid::TetraGrid(const coDoUnstructuredGrid *grid, const coDoVec3 *velocity,
                     const coDoFloat *volume, const coDoFloat *neighbor, int smode, int numtraces)
{
    // retrieve the data from the given objects
    int i;
    lastTimeNeighbor = new int[numtraces];
    for (i = 0; i < numtraces; i++)
        lastTimeNeighbor[i] = -1;
    searchMode = smode;
    BOX_INCREASE_FACTOR = coCoviseConfig::getInt("Module.TetraTrace.BoxIncreaseFactor", 100);
    // grid
    grid->getGridSize(&numElem, &numConn, &numPoints);
    grid->getAddresses(&elemList, &connList, &xCoord, &yCoord, &zCoord);

    // velocity
    velocity->getAddresses(&velU, &velV, &velW);

    // volume
    volume->getAddress(&volumeList);

    // neighbor
    neighbor->getAddress(&neighborList);
    neighborListComplete = 0;
    for (i = 0; i < numElem && !neighborListComplete; i++)
        if (neighborList[(i * 5) + 4] != -1)
            neighborListComplete = 1;

    // compute dimensions of our "searching-box"
    float max;
    //int i;
    max = fabs(volumeList[0]);
    for (i = 0; i < numElem; i++)
        if (fabs(volumeList[i]) > max)
            max = fabs(volumeList[i]);
    // sk 11.04.2001
    // cerr << "volume max: " << max << endl;

    edgeDim = powf(max, 1.0f / 3.0f);

    //cerr << "edgeDim: " << edgeDim << endl;

    edgeDim *= 2.0; // we have to experiment a little to find the best value

    // argh

    numTNCache = 0;
    TNCacheIndex = 0;

    // done
    return;
}

TetraGrid::~TetraGrid()
{
    // dummy
    delete[] lastTimeNeighbor;
}

coDoFloat *TetraGrid::buildNewNeighborList(char *name, TetraGrid *toGrid)
{
    coDoFloat *r = NULL;
    int i;

    if (name)
        r = new coDoFloat(name, numElem * 5, neighborList);
    else
        for (i = 0; i < numElem; i++)
            neighborList[(i * 5) + 4] = (float)getTimeNeighbor(i, toGrid, 0);

    return (r);
}

int TetraGrid::getTimeNeighbor(int el, TetraGrid *nGrid, int traceNum)
{
    int n;
    float x, y, z, waste;
    int o, c0, c1, c2, c3;
    int i;

#if !PRECOMP_TN
    // maybe we don't have to do anything
    if (neighborListComplete)
        return ((int)neighborList[(el * 5) + 4]);

    n = (int)neighborList[(el * 5) + 4];

    // speedup-check if we have the entire neighborlist
    //return( n ); //SHIT

    if (!nGrid || n != -1)
        return (n);

    // maybe we did cache 'em
    for (i = 0; i < numTNCache; i++)
        if (TNCache[i][0] == el)
            return (TNCache[i][1]);
#endif

    // argh
    o = elemList[el];
    c0 = connList[o];
    c1 = connList[o + 1];
    c2 = connList[o + 2];
    c3 = connList[o + 3];
    x = (xCoord[c0] + xCoord[c1] + xCoord[c2] + xCoord[c3]) / 4;
    y = (yCoord[c0] + yCoord[c1] + yCoord[c2] + yCoord[c3]) / 4;
    z = (zCoord[c0] + zCoord[c1] + zCoord[c2] + zCoord[c3]) / 4;

    n = nGrid->findActCell(x, y, z, waste, waste, waste, lastTimeNeighbor[traceNum]);
    if (n == -1 && lastTimeNeighbor[traceNum] != -1)
        n = nGrid->findActCell(x, y, z, waste, waste, waste, -1);

    //if( n!=-1 )
    lastTimeNeighbor[traceNum] = n;

// cache it
//neighborList[(el*5)+4] = (float)n;

#if !PRECOMP_TN
    TNCache[TNCacheIndex][0] = el;
    TNCache[TNCacheIndex][1] = n;
    TNCacheIndex++;
    if (numTNCache < 1000)
        numTNCache++;
    if (TNCacheIndex == 1000)
        TNCacheIndex = 0;
#endif

    //fprintf(stderr, "TetraGrid::getTimeNeighbor el=%d, n=%d\n", el, n);

    // done
    return (n);
}

float TetraGrid::tetraVolume(float p0[3], float p1[3], float p2[3], float p3[3])
{
    float v;

    v = (((p2[1] - p0[1]) * (p3[2] - p0[2]) - (p3[1] - p0[1]) * (p2[2] - p0[2])) * (p1[0] - p0[0]) + ((p2[2] - p0[2]) * (p3[0] - p0[0]) - (p3[2] - p0[2]) * (p2[0] - p0[0])) * (p1[1] - p0[1]) + ((p2[0] - p0[0]) * (p3[1] - p0[1]) - (p3[0] - p0[0]) * (p2[1] - p0[1])) * (p1[2] - p0[2])) / 6.0f;

    return (v);
}

float TetraGrid::computeSingleSide(float x, float y, float z, int c0, int c1, int c2, float u, float v, float w)
{
    // this function will return the time the particle will flow from the
    // given position with the given velocity until it hits the given side
    // of a tetrahedra.
    // it will return -1.0 if the ray is parallel to the plane or if
    // the particle has allready passed the plane and will therefore
    // never hit it

    /////
    ///// this algorithm is based on the one presented in
    /////    Graphic Gems I, pages 390pp
    /////

    // the ray
    float O[3], D[3];
    float t;

    // the plane
    float N[3], P[3];
    float a[3], b[3];
    float d;

    // the polygon/triangle
    float V[3][3];

    // other stuff
    float q;
    //float P[3];

    // the triangle-intersection
    float n, n_max;
    int i1, i2;
    float u0, u1, u2;
    float v0, v1, v2;
    float alpha, beta;

    // load polygon
    V[0][0] = xCoord[c0];
    V[0][1] = yCoord[c0];
    V[0][2] = zCoord[c0];
    V[1][0] = xCoord[c1];
    V[1][1] = yCoord[c1];
    V[1][2] = zCoord[c1];
    V[2][0] = xCoord[c2];
    V[2][1] = yCoord[c2];
    V[2][2] = zCoord[c2];

    // compute plane
    a[0] = V[1][0] - V[0][0];
    a[1] = V[1][1] - V[0][1];
    a[2] = V[1][2] - V[0][2];
    b[0] = V[2][0] - V[0][0];
    b[1] = V[2][1] - V[0][1];
    b[2] = V[2][2] - V[0][2];

    N[0] = a[1] * b[2] - a[2] * b[1];
    N[1] = a[2] * b[0] - a[0] * b[2];
    N[2] = a[0] * b[1] - a[1] * b[0];

    d = -(V[0][0] * N[0] + V[0][1] * N[1] + V[0][2] * N[2]);

    // compute ray
    O[0] = x;
    O[1] = y;
    O[2] = z;
    D[0] = u;
    D[1] = v;
    D[2] = w;

    // compute intersection ray/plane
    q = N[0] * D[0] + N[1] * D[1] + N[2] * D[2];

    // plane and ray may be parallel
    if (q == 0.0)
        return (-1.0);

    // compute point of intersection
    t = -(d + (N[0] * O[0] + N[1] * O[1] + N[2] * O[2])) / q;

    if (t < 0.0 && t > -0.00000005)
        t = 0.0;

    if (t < 0.0)
    {
        float way;

        way = fabsf(((u * u) + (v * v) + (w * w)) * t);
        if (way < 0.000001)
            t = 0.0;
        else
            return (-2.0);
    }

    // compute point of intersection
    P[0] = O[0] + D[0] * t;
    P[1] = O[1] + D[1] * t;
    P[2] = O[2] + D[2] * t;

    // compute best projection
    n_max = fabsf(N[0]);
    i1 = 1;
    i2 = 2;
    n = fabsf(N[1]);
    if (n > n_max)
    {
        n_max = n;
        i1 = 0;
        i2 = 2;
    }
    n = fabsf(N[2]);
    if (n > n_max)
    {
        n_max = n;
        i1 = 0;
        i2 = 1;
    }

    // compute projected points
    u0 = P[i1] - V[0][i1];
    u1 = V[1][i1] - V[0][i1];
    u2 = V[2][i1] - V[0][i1];

    v0 = P[i2] - V[0][i2];
    v1 = V[1][i2] - V[0][i2];
    v2 = V[2][i2] - V[0][i2];

    // and the solutions
    q = (u1 * v2) - (u2 * v1);
    alpha = (u0 * v2 - u2 * v0) / q;
    beta = (u1 * v0 - u0 * v1) / q;

    // is the given point inside the triangle
    if (!(alpha >= -0.000002 && beta >= -0.000002 && (alpha + beta) <= 1.0003))
        // no
        return (-3.0);

    // ok
    return (t);
}

int TetraGrid::leftWhichSide(int el, float &x, float &y, float &z, float u, float v, float w, float &dt)
{
    float r_min, r;
    int s, o;
    int c0, c1, c2, c3;

    float r0, r1, r2, r3;

    // a: point on plane
    // p: x/y/z

    o = elemList[el];
    c0 = connList[o];
    c1 = connList[o + 1];
    c2 = connList[o + 2];
    c3 = connList[o + 3];

    if (u == 0.0 && v == 0.0 && w == 0.0)
    {
        // as we have velocity 0 here, we will never ever
        // leave the cell
        return (-1);
    }

    // compute each single side
    r_min = 10000000.0; // this is a value we are likely to never exceed
    s = -1;

    // side 0 (012)
    r0 = r = computeSingleSide(x, y, z, c0, c1, c2, u, v, w);
    if (r >= 0.0 && r < r_min)
    {
        r_min = r;
        s = 0;
    }

    // side 1 (031)
    r1 = r = computeSingleSide(x, y, z, c0, c3, c1, u, v, w);
    if (r >= 0.0 && r < r_min)
    {
        r_min = r;
        s = 1;
    }

    // side 2 (023)
    r2 = r = computeSingleSide(x, y, z, c0, c2, c3, u, v, w);
    if (r >= 0.0 && r < r_min)
    {
        r_min = r;
        s = 2;
    }

    // side 3 (132)
    r3 = r = computeSingleSide(x, y, z, c1, c3, c2, u, v, w);
    if (r >= 0.0 && r < r_min)
    {
        r_min = r;
        s = 3;
    }

    // done
    dt = r_min;
    analyzeCase(r0, r1, r2, r3, dt, s, x, y, z, c0, c1, c2, c3);

    // debugging
    if (s == -1)
    {
        fprintf(stderr, "%d r: %19.16f %19.16f %19.16f %19.16f\n", el, r0, r1, r2, r3);
    }

    // done
    return (s);
}

void TetraGrid::analyzeCase(float t0, float t1, float t2, float t3, float &t, int &s, float &x, float &y, float &z, int c0, int c1, int c2, int c3)
{
    (void)c1;
    int numZero, numValid;
    float p[3];
    p[0] = p[1] = p[2] = 0.0f;

    // compute numZero/numValid
    numZero = numValid = 0;
    if (t0 == 0.0)
        numZero++;
    else if (t0 > 0.0)
        numValid++;

    if (t1 == 0.0)
        numZero++;
    else if (t1 > 0.0)
        numValid++;

    if (t2 == 0.0)
        numZero++;
    else if (t2 > 0.0)
        numValid++;

    if (t3 == 0.0)
        numZero++;
    else if (t3 > 0.0)
        numValid++;

    // now decide which case we have
    if (numZero == 3)
    {
        // on one vertex (corner)
        if (numValid == 1)
        {
            // entering the cell, fine -> C.1

            // find which way we leave
            if (t0 > 0.0)
            {
                t = t0;
                s = 0;
                return;
            }
            if (t1 > 0.0)
            {
                t = t1;
                s = 1;
                return;
            }
            if (t2 > 0.0)
            {
                t = t2;
                s = 2;
                return;
            }
            if (t3 > 0.0)
            {
                t = t3;
                s = 3;
                return;
            }
        }
        else
        {
            // leaving, the cell -> C.2 -> WC

            // we are on one vertex

            fprintf(stderr, "case C.2\n");

            t = 0.1f;
            s = -1;

            return;
        }
    }

    if (numZero == 2)
    {
        // on an edge
        if (numValid > 0)
        {
            // entering the cell -> A.1

            // find the "nearest" side
            s = -1;
            if (t0 > 0.0 && (t0 < t || s == -1))
            {
                s = 0;
                t = t0;
            }
            if (t1 > 0.0 && (t1 < t || s == -1))
            {
                s = 1;
                t = t1;
            }
            if (t2 > 0.0 && (t2 < t || s == -1))
            {
                s = 2;
                t = t2;
            }
            if (t3 > 0.0 && (t3 < t || s == -1))
            {
                s = 3;
                t = t3;
            }

            // done
            return;
        }
        else
        {
            // leaving the cell -> A.2 -> WC
            t = 0.0;

            // we are on an edge of the tetrahedra
            // check on which
            s = -1;
            if (t0 == 0.0)
            {
                // might be edges 01, 02 or 03
                // so anyway move a little bit towards c3
                p[0] = xCoord[c3];
                p[1] = yCoord[c3];
                p[2] = zCoord[c3];

                // we have to determine the side we will now pass by
                if (t1 == 0.0)
                    s = 1;
                else if (t2 == 0.0)
                    s = 2;
                else if (t3 == 0.0)
                    s = 3;
            }
            else if (t1 == 0.0)
            {
                // might be edges 12 or 13
                // towards c2
                p[0] = xCoord[c2];
                p[1] = yCoord[c2];
                p[2] = zCoord[c2];

                // which side ?
                if (t2 == 0.0)
                    s = 2;
                else if (t3 == 0.0)
                    s = 3;
            }
            else if (t2 == 0.0)
            {
                // it is edge 23
                p[0] = xCoord[c0];
                p[1] = yCoord[c0];
                p[2] = zCoord[c0];
                s = 2;
            }

            // compute "step"
            p[0] -= x;
            p[1] -= y;
            p[2] -= z;
            p[0] *= 0.00001f; // this may be adjusted due to any reason
            p[1] *= 0.00001f;
            p[2] *= 0.00001f;

            // perform step
            x += p[0];
            y += p[1];
            z += p[2];

            // tell caller to continue with neighbor
            t = 0.0;

            // done
            return;
        }
    }

    if (numZero == 1)
    {
        // on one side
        if (numValid > 0)
        {
            // entering cell -> B.1

            // find the "nearest" side
            s = -1;
            if (t0 > 0.0 && (t0 < t || s == -1))
            {
                s = 0;
                t = t0;
            }
            if (t1 > 0.0 && (t1 < t || s == -1))
            {
                s = 1;
                t = t1;
            }
            if (t2 > 0.0 && (t2 < t || s == -1))
            {
                s = 2;
                t = t2;
            }
            if (t3 > 0.0 && (t3 < t || s == -1))
            {
                s = 3;
                t = t3;
            }

            // done
            return;
        }
        else
        {
            // leaving cell -> B.2 , so we should continue with the neighbor here !
            // this is indicated by returning a stepSize of 0.0
            t = 0.0;

            if (t0 == 0.0)
            {
                s = 0;
                return;
            }
            if (t1 == 0.0)
            {
                s = 1;
                return;
            }
            if (t2 == 0.0)
            {
                s = 2;
                return;
            }
            if (t3 == 0.0)
            {
                s = 3;
                return;
            }
        }
    }

    // inside the cell, this is one of the D - cases
    if (numValid > 0)
    {
        // we don't differ between D-cases, as they might lead to one
        // of the WCs

        // so just find closest side hit
        s = -1;
        if (t0 > 0.0 && (t0 < t || s == -1))
        {
            s = 0;
            t = t0;
        }
        if (t1 > 0.0 && (t1 < t || s == -1))
        {
            s = 1;
            t = t1;
        }
        if (t2 > 0.0 && (t2 < t || s == -1))
        {
            s = 2;
            t = t2;
        }
        if (t3 > 0.0 && (t3 < t || s == -1))
        {
            s = 3;
            t = t3;
        }

        // done
        return;
    }
    else
    {
        fprintf(stderr, "undefined case: will never leave cell. This should be impossible !\n");
        fprintf(stderr, "                (but it might happen due to numerical inconsistencies !\n");
        s = -1;
        t = 0.1f;
    }

    // done
    return;
}

int TetraGrid::isInCell(int el, float x, float y, float z)
{
    float px[3], p0[3], p1[3], p2[3], p3[3];
    int o;
    int i;
    int r = -1;

    float w, w0, w1, w2, w3;

    // load coordinates
    px[0] = x;
    px[1] = y;
    px[2] = z;

    o = elemList[el];

    i = connList[o];
    p0[0] = xCoord[i];
    p0[1] = yCoord[i];
    p0[2] = zCoord[i];

    i = connList[o + 1];
    p1[0] = xCoord[i];
    p1[1] = yCoord[i];
    p1[2] = zCoord[i];

    i = connList[o + 2];
    p2[0] = xCoord[i];
    p2[1] = yCoord[i];
    p2[2] = zCoord[i];

    i = connList[o + 3];
    p3[0] = xCoord[i];
    p3[1] = yCoord[i];
    p3[2] = zCoord[i];

    w = fabsf(volumeList[el]);

    w0 = fabsf(tetraVolume(px, p1, p2, p3)) / w;
    w1 = fabsf(tetraVolume(p0, px, p2, p3)) / w;
    w2 = fabsf(tetraVolume(p0, p1, px, p3)) / w;
    w3 = fabsf(tetraVolume(p0, p1, p2, px)) / w;

    if (w0 + w1 + w2 + w3 <= 1.0001)
        r = el;

    // done
    return (r);
}

int TetraGrid::isInCell(int el, float x, float y, float z, float &u, float &v, float &w)
{
    float px[3], p0[3], p1[3], p2[3], p3[3];
    int o;
    int i;
    int r = -1;

    float wg, w0, w1, w2, w3;

    // load coordinates
    px[0] = x;
    px[1] = y;
    px[2] = z;

    o = elemList[el];

    i = connList[o];
    p0[0] = xCoord[i];
    p0[1] = yCoord[i];
    p0[2] = zCoord[i];

    i = connList[o + 1];
    p1[0] = xCoord[i];
    p1[1] = yCoord[i];
    p1[2] = zCoord[i];

    i = connList[o + 2];
    p2[0] = xCoord[i];
    p2[1] = yCoord[i];
    p2[2] = zCoord[i];

    i = connList[o + 3];
    p3[0] = xCoord[i];
    p3[1] = yCoord[i];
    p3[2] = zCoord[i];

    wg = fabsf(volumeList[el]);

    w0 = fabsf(tetraVolume(px, p1, p2, p3)) / wg;
    w1 = fabsf(tetraVolume(p0, px, p2, p3)) / wg;
    w2 = fabsf(tetraVolume(p0, p1, px, p3)) / wg;
    w3 = fabsf(tetraVolume(p0, p1, p2, px)) / wg;

    if (w0 + w1 + w2 + w3 <= 1.0001)
    {
        r = el;
        u = velU[connList[o]] * w0 + velU[connList[o + 1]] * w1 + velU[connList[o + 2]] * w2 + velU[connList[o + 3]] * w3;
        v = velV[connList[o]] * w0 + velV[connList[o + 1]] * w1 + velV[connList[o + 2]] * w2 + velV[connList[o + 3]] * w3;
        w = velW[connList[o]] * w0 + velW[connList[o + 1]] * w1 + velW[connList[o + 2]] * w2 + velW[connList[o + 3]] * w3;
    }

    // done
    return (r);
}

int TetraGrid::findActCell(float &x, float &y, float &z, float &u, float &v, float &w, int el)
{
    int r = -1;
    static float lastv = 0.0;

    if (el == -1) // this may be improved for better performance !!!
    {
        // overall search
        float bx0, by0, bz0;
        float bx1, by1, bz1;
        float xt, yt, zt;

        int i, o, c;

        // outer loop increases bounding box if we don't find start cells

        int factor_count = 1;

        float bbxWidthFactor = edgeDim;

        // cerr << "initial bounding box width: " << bbxWidthFactor << endl;

        while (factor_count < BOX_INCREASE_FACTOR && r == -1)
        {
            // sk: 11.04.2001
            // if(factor_count > 1)
            //   cerr << "increased bounding box[" << factor_count << "]: " << bbxWidthFactor << endl;

            // compute box
            bx0 = x - bbxWidthFactor;
            by0 = y - bbxWidthFactor;
            bz0 = z - bbxWidthFactor;
            bx1 = x + bbxWidthFactor;
            by1 = y + bbxWidthFactor;
            bz1 = z + bbxWidthFactor;

            // work through all cells to see if any of them contains the
            // given point
            for (i = 0; i < numElem && r == -1; i++)
            {
                o = elemList[i];
                c = connList[o];

                xt = xCoord[c];
                yt = yCoord[c];
                zt = zCoord[c];

                if ((bx0 < xt && xt < bx1) && (by0 < yt && yt < by1) && (bz0 < zt && zt < bz1))
                {
                    // this cell is inside the box, so check if the point
                    // is also
                    r = isInCell(i, x, y, z, u, v, w);

                    // maybe trace to the correct cell from this cell ?!?
                }
            }
            factor_count++;
            bbxWidthFactor *= 2;
        }
        // done
    }
    else
    {
        // perform a trace

        // maybe we didn't leave the cell
        if ((r = isInCell(el, x, y, z, u, v, w)) == -1)
        {
            // we left it
            float xA, yA, zA; // actual position of trace
            float xV, yV, zV; // vector from actual position to
            //  target position (x,y,z)
            int cA;

            int o, c0, c1, c2, c3;
            int s, numPassed;
            float dt;

            // we start at the centroid of the source cell
            o = elemList[el];
            c0 = connList[o];
            c1 = connList[o + 1];
            c2 = connList[o + 2];
            c3 = connList[o + 3];

            xA = (xCoord[c0] + xCoord[c1] + xCoord[c2] + xCoord[c3]) / 4.0f;
            yA = (yCoord[c0] + yCoord[c1] + yCoord[c2] + yCoord[c3]) / 4.0f;
            zA = (zCoord[c0] + zCoord[c1] + zCoord[c2] + zCoord[c3]) / 4.0f;
            cA = el;

            xV = x - xA;
            yV = y - yA;
            zV = z - zA;

            s = 0;
            numPassed = 0;
            while (s >= 0 && r == -1 && numPassed < 700) // warum 700
            {
                s = leftWhichSide(cA, xA, yA, zA, xV, yV, zV, dt);

                if (s >= 0)
                {
                    // we left cA through side s
                    // so check if there is a neighbor on that side
                    cA = (int)neighborList[(cA * 5) + s];
                    if (cA < 0)
                    {
                        // no neighbor
                        if (searchMode == 2)
                        {
                            // Uwe Woessner
                            // could be in another block, so try an overall search

                            // overall search
                            float bx0, by0, bz0;
                            float bx1, by1, bz1;
                            float xt, yt, zt;

                            int i, o, c;

                            // compute box
                            bx0 = x - edgeDim;
                            by0 = y - edgeDim;
                            bz0 = z - edgeDim;
                            bx1 = x + edgeDim;
                            by1 = y + edgeDim;
                            bz1 = z + edgeDim;

                            // work through all cells to see if any of them contains the
                            // given point
                            for (i = 0; i < numElem && r == -1; i++)
                            {
                                o = elemList[i];
                                c = connList[o];

                                xt = xCoord[c];
                                yt = yCoord[c];
                                zt = zCoord[c];

                                if ((bx0 < xt && xt < bx1) && (by0 < yt && yt < by1) && (bz0 < zt && zt < bz1))
                                {
                                    // this cell is inside the box, so check if the point
                                    // is also
                                    r = isInCell(i, x, y, z, u, v, w);

                                    // maybe trace to the correct cell from this cell ?!?
                                }
                            }
                        }
                        s = -1;
                    }
                    else
                    {
                        // neighbor exists
                        if ((r = isInCell(cA, x, y, z, u, v, w)) == -1)
                        {
                            // the point isn't in the neighbor, so we
                            // have to continue tracing
                            o = elemList[cA];
                            c0 = connList[o];
                            c1 = connList[o + 1];
                            c2 = connList[o + 2];
                            c3 = connList[o + 3];

                            xA = (xCoord[c0] + xCoord[c1] + xCoord[c2] + xCoord[c3]) / 4.0f;
                            yA = (yCoord[c0] + yCoord[c1] + yCoord[c2] + yCoord[c3]) / 4.0f;
                            zA = (zCoord[c0] + zCoord[c1] + zCoord[c2] + zCoord[c3]) / 4.0f;
                            //cA = el;

                            xV = x - xA;
                            yV = y - yA;
                            zV = z - zA;
                            float v = xV * xV + yV * yV + zV * zV;
                            if (lastv < v)
                            {
                                // ok
                                numPassed++;
                            }
                            lastv = v;
                        }
                    }
                }
            }

            // NOTE: s==-1 may also indicate that the velocity is 0
        }

        // done
    }

    // done
    return (r);
}
