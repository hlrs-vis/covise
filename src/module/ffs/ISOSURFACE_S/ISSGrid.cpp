/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ISSGrid.h"

ISSGrid::ISSGrid(coDistributedObject *grid, coDistributedObject *data)
{
    char *dataType;

    // check data
    validFlag = 0;
    if (!grid || !data)
        return;
    dataType = grid->getType();
    if (!strcmp(dataType, "UNIGRD"))
    {
        // ok
        gridPtr = grid;
        gridType = UNIGRD;
    }
    else
        return; // not supported
    dataType = data->getType();
    if (!strcmp(dataType, "STRSDT"))
    {
        // ok
        coDoFloat *dataIn = (coDoFloat *)data;
        dataIn->getAddress(&dataPtr);
        dataIn->getGridSize(&numI, &numJ, &numK);
    }
    else
        return; // not supported

    // all right
    validFlag = 1;

    // done
    return;
}

ISSGrid::~ISSGrid()
{
}

void ISSGrid::getSlice(int i, float *x, float *y, float *z, float *d)
{
    // NOTE: x, y, z, d must be allocated/freed elsewhere

    int o, j, k;
    float *xPtr, *yPtr, *zPtr, *dPtr;
    if (gridType == UNIGRD)
    {
        // fast
        coDoUniformGrid *g = (coDoUniformGrid *)gridPtr;
        float dx, dy, dz, xMin, yMin, zMin, w;
        float lX, lY, lZ; // local values
        g->getDelta(&dx, &dy, &dz);
        g->getMinMax(&xMin, &w, &yMin, &w, &zMin, &w);

        lX = xMin + ((float)i) * dx;
        lY = yMin;
        o = i * numJ * numK;
        xPtr = x;
        yPtr = y;
        zPtr = z;
        dPtr = d;
        for (j = 0; j < numJ; j++)
        {
            lZ = zMin;
            for (k = 0; k < numK; k++)
            {
                *xPtr = lX;
                *yPtr = lY;
                *zPtr = lZ;
                *dPtr = dataPtr[o];
                // one step forward
                lZ += dz;
                o++;
                xPtr++;
                yPtr++;
                zPtr++;
                dPtr++;
            }
            lY += dy;
        }
    }

    return;
}

void ISSGrid::getDimensions(int *i, int *j, int *k)
{
    *i = numI;
    *j = numJ;
    *k = numK;
    return;
}
