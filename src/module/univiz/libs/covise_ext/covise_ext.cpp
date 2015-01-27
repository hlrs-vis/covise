/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// helpers for Covise
// filip sadlo
// cgl eth 2007

#include <vector>

#include "covise_ext.h"

using namespace covise;

void adaptMyParams(void)
{
    for (int i = 0; i < (int)myParams.size(); i++)
    {
        myParams[i]->adaptValue();
    }
}

coDoUnstructuredGrid *generateUniformUSG(const char *name,
                                         float originX, float originY, float originZ,
                                         int cellsX, int cellsY, int cellsZ,
                                         float cellSize)
{ // generates uniform hexahedral USG

    bool success = true;

    int ncells = cellsX * cellsY * cellsZ;
    int nnodes = (cellsX + 1) * (cellsY + 1) * (cellsZ + 1);

    coDoUnstructuredGrid *grid = NULL;

    int *elemList = new int[ncells];
    int *cornerList = new int[ncells * 8];
    float *coordX = new float[nnodes];
    float *coordY = new float[nnodes];
    float *coordZ = new float[nnodes];
    int *typeList = new int[ncells];
    if (!elemList || !cornerList || !coordX || !coordY || !coordZ || !typeList)
    {
        printf("allocation failed\n");
        success = false;
        goto FreeOut;
    }

    // set coordinates and connectivity
    {
        float *xp, *yp, *zp;
        xp = coordX;
        yp = coordY;
        zp = coordZ;
        int n = 0;
        int c = 0;
        for (int z = 0; z < cellsZ + 1; z++)
        {
            for (int y = 0; y < cellsY + 1; y++)
            {
                for (int x = 0; x < cellsX + 1; x++)
                {
                    *(xp++) = originX + x * cellSize;
                    *(yp++) = originY + y * cellSize;
                    *(zp++) = originZ + z * cellSize;

                    if ((x < cellsX) && (y < cellsY) && (z < cellsZ))
                    {
                        int nX = cellsX + 1;
                        int nXY = nX * (cellsY + 1);
                        int nXYPX = nXY + nX;
                        int list[8] = {
                            n, n + 1, n + nX + 1, n + nX,
                            n + nXY, n + nXY + 1, n + nXYPX + 1, n + nXYPX
                        };

                        memcpy(cornerList + c * 8, list, 8 * sizeof(int));
                        elemList[c] = c * 8;
                        typeList[c] = TYPE_HEXAEDER;

                        c++;
                    }
                    n++;
                }
            }
        }
    }

    grid = new coDoUnstructuredGrid(name, ncells, ncells * 8, nnodes, elemList,
                                    cornerList, coordX, coordY, coordZ, typeList);
    // ####### TODO: obj_ok() does not exist e.g. at VATECH
    //if (!grid->objectOk()) {
    if (!grid)
    {
        printf("generateUniformUSG: failed to create grid\n");
        success = false;
    }

FreeOut:
    if (elemList)
        delete[] elemList;
    if (cornerList)
        delete[] cornerList;
    if (coordX)
        delete[] coordX;
    if (coordY)
        delete[] coordY;
    if (coordZ)
        delete[] coordZ;
    if (typeList)
        delete[] typeList;

    if (success)
        return grid;
    else
        return NULL;
}
