/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ISSUtil.h"
#include "ISSGrid.h"

#include <string.h>

#include <iostream.h>

ISS3DSlice::ISS3DSlice(ISSGrid *g)
{
    grid = g;
    int t, i;

    // alloc mem
    grid->getDimensions(&numI, &numJ, &numK);
    t = numJ * numK;
    x1 = new float[t];
    y1 = new float[t];
    z1 = new float[t];
    d1 = new float[t];
    x2 = new float[t];
    y2 = new float[t];
    z2 = new float[t];
    d2 = new float[t];
    t = (numJ - 1) * (numK - 1) * 8;
    vl1 = new int[t];
    vl2 = new int[t];
    vlInter = new int[(numJ - 1) * (numK - 1) * 20];

    // prepare first slice
    curSlice = 1;
    grid->getSlice(0, x1, y1, z1, d1);
    grid->getSlice(1, x2, y2, z2, d2);
    for (i = 0; i < t; i++)
    {
        vl1[i] = -2; // -2: not computed, -1: unused
        vl2[i] = -2;
    }
    t = (numJ - 1) * (numK - 1) * 20;
    for (i = t; i < t; i++)
        vlInter[i] = -2;
    curUpper = 2;

    // done
    return;
}

ISS3DSlice::~ISS3DSlice()
{
    // clean up
    delete[] x1;
    delete[] y1;
    delete[] z1;
    delete[] d1;
    delete[] x2;
    delete[] y2;
    delete[] z2;
    delete[] d2;
    delete[] vl1;
    delete[] vl2;
    delete[] vlInter;

    // done
    return;
}

void ISS3DSlice::sliceUp()
{
    int i, t;

    t = (numJ - 1) * (numK - 1) * 8;

    if (curUpper == 2)
    {
        // replace slice1
        curSlice++;
        grid->getSlice(curSlice, x1, y1, z1, d1);

        for (i = 0; i < t; i++)
            vl1[i] = -2;

        curUpper = 1; // slice1 is new upper slice
    }
    else
    {
        // replace slice2
        curSlice++;
        grid->getSlice(curSlice, x2, y2, z2, d2);

        for (i = 0; i < t; i++)
            vl2[i] = -2;

        curUpper = 2; // slice2 is new upper slice
    }

    // inter
    t = (numJ - 1) * (numK - 1) * 20;
    for (i = t; i < t; i++)
        vlInter[i] = -2;

    // done
    return;
}

void ISS3DSlice::handleSlice(int *vertList, int *numVert, int *stripList, int *numStrips,
                             float *xCoord, float *yCoord, float *zCoord, float threshold)
{
    //int j, k;
    //int maxVert;

    // maxVert = vertList[(*numVert)-1];

    // walk the 2D-slices
    // lower [only has to be done if curSlice==1 (lowest slice)]
    if (curSlice == 1)
        handle2DSlice(vl1, d1, threshold);

    // upper
    if (curUpper == 2)
        handle2DSlice(vl2, d2, threshold);
    else
        handle2DSlice(vl1, d1, threshold);

    // and 3D

    // compute output (coordinates/stuff)

    cerr << ".";

    // done
    return;
}

void ISS3DSlice::handle2DSlice(int *vl, float *d, float th)
{
    int j, k, o, co, o2;
    //int i;
    float A, B, C, D, E;
    int vertOffs;

    vertOffs = 0;

    for (j = 0; j < numJ - 1; j++)
    {
        o = (numK - 1) * j * 8;
        for (k = 0; k < numK - 1; k++)
        {
            co = o + (k * 8);

            // handle current quad
            o2 = j * numK;
            A = d[o2 + k];
            B = d[o2 + k + 1];
            o2 += numK;
            C = d[o2 + k];
            D = d[o2 + k + 1];
            E = (A + B + C + D) / 4.0;

            // 0
            if (vl[co] == -2)
            {
                if ((D > th && A < th) || (D < th && A > th))
                {
                    // intersected
                    vl[co] = vertOffs;
                    vertOffs++;
                }
            }

            // 1
            if (vl[co + 1] == -2)
            {
                if ((C > th && D < th) || (C < th && D > th))
                {
                    // intersected
                    vl[co + 1] = vertOffs;
                    vertOffs++;
                }
            }

            // 2
            if (vl[co + 2] == -2)
            {
                if ((B > th && C < th) || (B < th && C > th))
                {
                    // intersected
                    vl[co + 2] = vertOffs;
                    vertOffs++;
                }
            }

            // 3
            if (vl[co + 3] == -2)
            {
                if ((A > th && B < th) || (A < th && B > th))
                {
                    // intersected
                    vl[co + 3] = vertOffs;
                    vertOffs++;
                }
            }

            // 4
            if (vl[co + 4] == -2)
            {
                if ((E > th && D < th) || (E < th && D > th))
                {
                    // intersected
                    vl[co + 4] = vertOffs;
                    vertOffs++;
                }
            }

            // 5
            if (vl[co + 5] == -2)
            {
                if ((B > th && E < th) || (B < th && E > th))
                {
                    // intersected
                    vl[co + 5] = vertOffs;
                    vertOffs++;
                }
            }

            // 6
            if (vl[co + 6] == -2)
            {
                if ((E > th && A < th) || (E < th && A > th))
                {
                    // intersected
                    vl[co + 6] = vertOffs;
                    vertOffs++;
                }
            }

            // 7
            if (vl[co + 7] == -2)
            {
                if ((C > th && E < th) || (C < th && E > th))
                {
                    // intersected
                    vl[co + 7] = vertOffs;
                    vertOffs++;
                }
            }
        }
    }

    // done
    return;
}
