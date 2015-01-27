/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <appl/ApplInterface.h>

#include "IsoSurfaceS.h"
#include "ISSGrid.h"
#include "ISSUtil.h"
#include "ISSTables.h"

#include <iostream.h>
#include <string.h>

int main(int argc, char *argv[])
{
    // init
    IsoSurfaceS *appl = new IsoSurfaceS(argc, argv);

    // go
    appl->run();

    // done
    delete appl;
    return (0);
}

IsoSurfaceS::~IsoSurfaceS()
{
    // clean up

    return;
}

coDistributedObject **IsoSurfaceS::compute(coDistributedObject **inObj, char **outNames)
{
    coDistributedObject **returnObject = NULL;
    char *dataType;

    // get param
    Covise::get_scalar_param("isoValue", &isoValue);
    Covise::get_choice_param("algorithm", &algorithm);
    Covise::get_vector_param("isopoint", 0, isoPoint);
    Covise::get_vector_param("isopoint", 1, isoPoint + 1);
    Covise::get_vector_param("isopoint", 2, isoPoint + 2);

    // check params/data
    dataType = inObj[0]->getType();
    if (strcmp(dataType, "UNIGRD"))
    {
        // error
        Covise::sendError("IsoSurfaceS requires UNIGRD");
        return (NULL);
    }

    // prepare output
    returnObject = new coDistributedObject *[3];
    returnObject[0] = NULL;
    returnObject[1] = NULL;
    returnObject[2] = NULL;

    // and go
    switch (algorithm)
    {
    case 1:
        returnObject[0] = issVoxels((coDoUniformGrid *)inObj[0], (coDoFloat *)inObj[1], outNames[0]);
        break;
    case 3:
        returnObject[0] = issSkeletonClimbing((coDoUniformGrid *)inObj[0], (coDoFloat *)inObj[1], outNames[0]);
        break;
    default:
        returnObject[0] = issMCubes((coDoUniformGrid *)inObj[0], (coDoFloat *)inObj[1], outNames[0]);
        break;
    }

    // done
    return (returnObject);
}

coDistributedObject *IsoSurfaceS::issVoxels(coDoUniformGrid *grid, coDoFloat *data, char *outName)
{
    // output vars
    coDoTriangleStrips *triStrips = NULL;
    int numStrips, numVert, numCoord;
    float *xCoord, *yCoord, *zCoord;
    int *vertList, *stripList;

    // input vars
    float *dataPtr = NULL;
    int numI, numJ, numK;
    float dx, dy, dz, xMin, yMin, zMin;

    // own stuff
    char *voxels = NULL;
    char *voxelPtr;
    int voxelUpDown, voxelLeftRight, curVoxel;
    int i, j, k;

    // get input
    grid->getMinMax(&xMin, &dx, &yMin, &dy, &zMin, &dz);
    grid->getDelta(&dx, &dy, &dz);
    grid->getGridSize(&numI, &numJ, &numK);
    data->getAddress(&dataPtr);

    // prepare for hyperspeed
    voxels = new char[numI * numJ * numK];
    voxelUpDown = numK;
    voxelLeftRight = numJ * numK;
    for (i = 0; i < numI * numJ * numK; i++)
    {
        if (dataPtr[i] >= isoValue)
            voxels[i] = 0x80;
        else
            voxels[i] = 0x40;
    }

    // first run...determine output size
    numStrips = 0;
    numVert = 0;
    numCoord = 0;
    // boundary voxels

    // we don't care 'bout the lame boundary voxels

    // inner voxels
    for (i = 1; i < numI - 1; i++)
    {
        for (j = 1; j < numJ - 1; j++)
        {
            curVoxel = i * voxelLeftRight + j * voxelUpDown + 1;
            for (k = 1; k < numK - 1; k++)
            {
                voxelPtr = voxels + curVoxel;

                if ((int)((*voxelPtr | voxels[curVoxel + voxelLeftRight]) & 0xC0) == 0xC0)
                {
                    *voxelPtr |= 4;
                    numStrips++;
                    numVert += 4;
                    numCoord += 4;
                }

                if ((int)((*voxelPtr | voxels[curVoxel - 1]) & 0xC0) == 0xC0)
                {
                    *voxelPtr |= 2;
                    numStrips++;
                    numVert += 4;
                    numCoord += 4;
                }
                if ((int)((*voxelPtr | voxels[curVoxel - voxelUpDown]) & 0xC0) == 0xC0)
                {
                    *voxelPtr |= 32;
                    numStrips++;
                    numVert += 4;
                    numCoord += 4;
                }

                curVoxel++;
            }
        }
    }

    // create object
    triStrips = new coDoTriangleStrips(outName, numCoord, numVert, numStrips);
    triStrips->getAddresses(&xCoord, &yCoord, &zCoord, &vertList, &stripList);
    triStrips->addAttribute("vertexOrder", "2");

    numStrips = 0;
    numVert = 0;
    numCoord = 0;
    for (i = 1; i < numI - 1; i++)
    {
        for (j = 1; j < numJ - 1; j++)
        {
            curVoxel = i * voxelLeftRight + j * voxelUpDown + 1;
            for (k = 1; k < numK - 1; k++)
            {
                voxelPtr = voxels + curVoxel;
                if (*voxelPtr & 4)
                {
                    // right
                    stripList[numStrips] = numVert;
                    numStrips++;
                    xCoord[numCoord] = xMin + ((float)i + 1) * dx;
                    yCoord[numCoord] = yMin + ((float)j) * dy;
                    zCoord[numCoord] = zMin + ((float)k) * dz;
                    xCoord[numCoord + 1] = xMin + ((float)i + 1) * dx;
                    yCoord[numCoord + 1] = yMin + ((float)j + 1) * dy;
                    zCoord[numCoord + 1] = zMin + ((float)k) * dz;
                    xCoord[numCoord + 2] = xMin + ((float)i + 1) * dx;
                    yCoord[numCoord + 2] = yMin + ((float)j + 1) * dy;
                    zCoord[numCoord + 2] = zMin + ((float)k + 1) * dz;
                    xCoord[numCoord + 3] = xMin + ((float)i + 1) * dx;
                    yCoord[numCoord + 3] = yMin + ((float)j) * dy;
                    zCoord[numCoord + 3] = zMin + ((float)k + 1) * dz;

                    vertList[numVert] = numCoord;
                    vertList[numVert + 1] = numCoord + 1;
                    vertList[numVert + 2] = numCoord + 3;
                    vertList[numVert + 3] = numCoord + 2;

                    numCoord += 4;
                    numVert += 4;
                }
                if (*voxelPtr & 2)
                {
                    // behind
                    stripList[numStrips] = numVert;
                    numStrips++;
                    xCoord[numCoord] = xMin + ((float)i) * dx;
                    yCoord[numCoord] = yMin + ((float)j) * dy;
                    zCoord[numCoord] = zMin + ((float)k) * dz;
                    xCoord[numCoord + 1] = xMin + ((float)i) * dx;
                    yCoord[numCoord + 1] = yMin + ((float)j + 1) * dy;
                    zCoord[numCoord + 1] = zMin + ((float)k) * dz;
                    xCoord[numCoord + 2] = xMin + ((float)i + 1) * dx;
                    yCoord[numCoord + 2] = yMin + ((float)j + 1) * dy;
                    zCoord[numCoord + 2] = zMin + ((float)k) * dz;
                    xCoord[numCoord + 3] = xMin + ((float)i + 1) * dx;
                    yCoord[numCoord + 3] = yMin + ((float)j) * dy;
                    zCoord[numCoord + 3] = zMin + ((float)k) * dz;

                    vertList[numVert] = numCoord;
                    vertList[numVert + 1] = numCoord + 1;
                    vertList[numVert + 2] = numCoord + 3;
                    vertList[numVert + 3] = numCoord + 2;

                    numCoord += 4;
                    numVert += 4;
                }
                if (*voxelPtr & 32)
                {
                    // below
                    stripList[numStrips] = numVert;
                    numStrips++;
                    xCoord[numCoord] = xMin + ((float)i) * dx;
                    yCoord[numCoord] = yMin + ((float)j) * dy;
                    zCoord[numCoord] = zMin + ((float)k + 1) * dz;
                    xCoord[numCoord + 1] = xMin + ((float)i) * dx;
                    yCoord[numCoord + 1] = yMin + ((float)j) * dy;
                    zCoord[numCoord + 1] = zMin + ((float)k) * dz;
                    xCoord[numCoord + 2] = xMin + ((float)i + 1) * dx;
                    yCoord[numCoord + 2] = yMin + ((float)j) * dy;
                    zCoord[numCoord + 2] = zMin + ((float)k) * dz;
                    xCoord[numCoord + 3] = xMin + ((float)i + 1) * dx;
                    yCoord[numCoord + 3] = yMin + ((float)j) * dy;
                    zCoord[numCoord + 3] = zMin + ((float)k + 1) * dz;

                    vertList[numVert] = numCoord;
                    vertList[numVert + 1] = numCoord + 1;
                    vertList[numVert + 2] = numCoord + 3;
                    vertList[numVert + 3] = numCoord + 2;

                    numCoord += 4;
                    numVert += 4;
                }

                curVoxel++;
            }
        }
    }

    cerr << numStrips << "  " << numVert << endl;

    // done
    return (triStrips);
    //return( pObj );
}

void IsoSurfaceS::fillRecursive(char *newNodes, char *oldNodes, int numI, int numJ, int numK, int i, int j, int k)
{
    int o;

    if (newNodes[i * numJ * numK + j * numK + k])
    {
        cerr << "E";
        return;
    }

    newNodes[i * numJ * numK + j * numK + k] = 1;
    if (i < numI - 1)
    {
        o = (i + 1) * numJ * numK + j * numK + k;
        if (oldNodes[o] && !newNodes[o])
            fillRecursive(newNodes, oldNodes, numI, numJ, numK, i + 1, j, k);
    }
    if (i > 0)
    {
        o = (i - 1) * numJ * numK + j * numK + k;
        if (oldNodes[o] && !newNodes[o])
            fillRecursive(newNodes, oldNodes, numI, numJ, numK, i - 1, j, k);
    }

    if (j < numJ - 1)
    {
        o = i * numJ * numK + (j + 1) * numK + k;
        if (oldNodes[o] && !newNodes[o])
            fillRecursive(newNodes, oldNodes, numI, numJ, numK, i, j + 1, k);
    }
    if (j > 0)
    {
        o = i * numJ * numK + (j - 1) * numK + k;
        if (oldNodes[o] && !newNodes[o])
            fillRecursive(newNodes, oldNodes, numI, numJ, numK, i, j - 1, k);
    }

    if (k < numK - 1)
    {
        o = i * numJ * numK + j * numK + k + 1;
        if (oldNodes[o] && !newNodes[o])
            fillRecursive(newNodes, oldNodes, numI, numJ, numK, i, j, k + 1);
    }
    if (k > 0)
    {
        o = i * numJ * numK + j * numK + k - 1;
        if (oldNodes[o] && !newNodes[o])
            fillRecursive(newNodes, oldNodes, numI, numJ, numK, i, j, k - 1);
    }

    //cerr << ".";
}

coDistributedObject *IsoSurfaceS::issMCubes(coDoUniformGrid *grid, coDoFloat *data, char *outName)
{
    // output vars
    coDoPolygons *poly = NULL;
    int numPoly, numVert, numCoord;
    float *xCoord, *yCoord, *zCoord;
    int *vertList, *polyList;

    // input vars
    float *dataPtr = NULL;
    int numI, numJ, numK;
    float dx, dy, dz, xMin, yMin, zMin;
    float xMax, yMax, zMax;

    // own stuff
    char *cells = NULL;
    char *nodes = NULL;
    int numHandle;
    //char *voxelPtr;
    //int voxelUpDown, voxelLeftRight, curVoxel;
    int i, j, k, t, o, a, b;
    int leftRight, upDown;
    float *floatPtr;
    char *cellsPtr, *nodesPtr;
    int *resortPtr;
    typedef struct cellInfo_s
    {
        int i, j, k; // position of cell
        char bitmap;
        int vl[12]; // vertices
        int nl[6]; // neighbor cells
        char vertOK; // set if vl[] valid
    } cellInfo;
    cellInfo *cI;
    int *resortTable;
    //   char bm, b1, b2, b3, b4, b5, b6, b7, b8;
    float localData[8], localCoord[8][3];
    float cellVal;
    int *polyNodes;

    // get input
    grid->getMinMax(&xMin, &xMax, &yMin, &yMax, &zMin, &zMax);
    grid->getGridSize(&numI, &numJ, &numK);
    //grid->getDelta( &dx, &dy, &dz );  // BUGGY !!!
    dx = (xMax - xMin) / (numI - 1);
    dy = (yMax - yMin) / (numJ - 1);
    dz = (zMax - zMin) / (numK - 1);

    data->getAddress(&dataPtr);
    leftRight = numJ * numK;
    upDown = numK;

    // first run, determine number of cells to handle
    cells = new char[(numI - 1) * (numJ - 1) * (numK - 1)];
    nodes = new char[numI * numJ * numK];
    resortTable = new int[(numI - 1) * (numJ - 1) * (numK - 1)];
    floatPtr = dataPtr;
    nodesPtr = nodes;
    for (i = 0; i < numI * numJ * numK; i++)
    {
        if (*floatPtr >= isoValue)
            *nodesPtr = 1;
        else
            *nodesPtr = 0;
        floatPtr++;
        nodesPtr++;
    }

    // maybe we have to do some filtering
    i = (int)((isoPoint[0] - xMin) / dx);
    j = (int)((isoPoint[1] - yMin) / dy);
    k = (int)((isoPoint[2] - zMin) / dz);

    // /*  vorerst mal disabled, irgendwo komischer bug
    //   der bug war nicht hier sondern in
    if (i >= 0 && j >= 0 && k >= 0)
    {
        // check
        if (nodes[i * numJ * numK + j * numK + k])
        {

            cerr << "Filtering from i j k :  " << i << " " << j << " " << k << endl;

            char *newNodes = new char[numI * numJ * numK];
            memset(newNodes, 0, numI * numJ * numK);

            fillRecursive(newNodes, nodes, numI, numJ, numK, i, j, k);

            delete[] nodes;
            nodes = newNodes;

            // start filtering from this cell
            //memset(nodes, 0, (numJ-1)*(numK-1)*(numI-1));
        }
    }
    //  */

    // go on
    numHandle = 0;
    t = (numJ - 1) * (numK - 1);
    for (i = 0; i < numI - 1; i++)
    {
        for (j = 0; j < numJ - 1; j++)
        {
            o = (i * t) + (j * (numK - 1));
            cellsPtr = cells + o;
            nodesPtr = nodes + (i * leftRight) + (j * upDown);
            resortPtr = resortTable + o;
            for (k = 0; k < numK - 1; k++)
            {
                // NOTE: covise-style node numbering used in documentation
                // 6
                *cellsPtr = (*nodesPtr) << 5;
                // 5
                *cellsPtr |= (*(nodesPtr + 1)) << 4;
                // 7
                nodesPtr += leftRight;
                *cellsPtr |= (*nodesPtr) << 6;
                // 8
                *cellsPtr |= (*(nodesPtr + 1)) << 7;
                // 3
                nodesPtr += upDown;
                *cellsPtr |= (*nodesPtr) << 2;
                // 4
                *cellsPtr |= (*(nodesPtr + 1)) << 3;
                // 2
                nodesPtr -= leftRight;
                *cellsPtr |= (*nodesPtr) << 1;
                // 1
                *cellsPtr |= (*(nodesPtr + 1));
                nodesPtr -= upDown;

                // does this one count ?
                if ((int)*cellsPtr == 0xFF || (int)*cellsPtr == 0x00)
                {
                    *cellsPtr = 0; // no
                    *resortPtr = -1;
                }
                else
                {
                    *resortPtr = numHandle;
                    numHandle++; // yes
                }

                // on to the next
                nodesPtr++;
                cellsPtr++;
                resortPtr++;
            }
        }
    }

    // make some room
    delete[] nodes;

    // some info on performance (testing)
    //cerr << "numHandle=" << numHandle << endl;
    //cerr << "total: " << t*(numI-1) << endl;

    // face-2-face neighbors are:
    //    0  left
    //    1  behind
    //    2  right
    //    3  before
    //    4  up
    //    5  down

    // vertices are on the following edges of each cube:
    //    0  1-2
    //    1  2-3
    //    2  3-4
    //    3  4-1
    //    4  5-6
    //    5  6-7
    //    6  7-8
    //    7  8-5
    //    8  1-5
    //    9  2-6
    //   10  3-7
    //   11  4-8

    // crunch crunch
    cI = new cellInfo[numHandle];
    if (!cI)
    {
        // not enough memory
        Covise::sendError("not enough memory");
        return (NULL);
    }
    numHandle = 0;
    upDown = numK - 1;
    leftRight = (numK - 1) * (numJ - 1);
    for (i = 0; i < numI - 1; i++)
    {
        for (j = 0; j < numJ - 1; j++)
        {
            o = (i * t) + (j * (numK - 1));
            cellsPtr = cells + o;
            for (k = 0; k < numK - 1; k++)
            {
                if (*cellsPtr)
                {
                    // use it
                    cI[numHandle].i = i;
                    cI[numHandle].j = j;
                    cI[numHandle].k = k;
                    cI[numHandle].bitmap = *cellsPtr;
                    cI[numHandle].vertOK = 0;
                    resortPtr = cI[numHandle].nl;
                    if (!i)
                        *resortPtr = -1;
                    else
                        *resortPtr = resortTable[o - leftRight];
                    resortPtr++;
                    if (!k)
                        *resortPtr = -1;
                    else
                        *resortPtr = resortTable[o - 1];
                    resortPtr++;
                    if (i == numI - 2)
                        *resortPtr = -1;
                    else
                        *resortPtr = resortTable[o + leftRight];
                    resortPtr++;
                    if (k == numK - 2)
                        *resortPtr = -1;
                    else
                        *resortPtr = resortTable[o + 1];
                    resortPtr++;
                    if (j == numJ - 2)
                        *resortPtr = -1;
                    else
                        *resortPtr = resortTable[o + upDown];
                    resortPtr++;
                    if (!j)
                        *resortPtr = -1;
                    else
                        *resortPtr = resortTable[o - upDown];

                    // done
                    numHandle++;
                }
                cellsPtr++;
                o++;
            }
        }
    }

    // temp. clean up
    delete[] resortTable;
    delete[] cells;

    // prepare tables
    computeMCubesTbl();

    // determine size of output
    numPoly = 0;
    numCoord = 0;
    leftRight = numJ * numK;
    upDown = numK;
    for (i = 0; i < numHandle; i++)
    {
        switch (mCubesTbl[cI[i].bitmap].caseSel)
        {
        case 1:
            numPoly++;
            break;
        case 2:
            numPoly += 2;
            break;
        case 3:
            o = cI[i].i * leftRight + cI[i].j * upDown + cI[i].k;
            localData[0] = dataPtr[o + upDown + 1];
            localData[1] = dataPtr[o + upDown];
            localData[2] = dataPtr[o + upDown + leftRight];
            localData[3] = dataPtr[o + upDown + leftRight + 1];
            localData[4] = dataPtr[o + 1];
            localData[5] = dataPtr[o];
            localData[6] = dataPtr[o + leftRight];
            localData[7] = dataPtr[o + leftRight + 1];
            polyNodes = mCubesTbl[cI[i].bitmap].nodePairs;
            cellVal = localData[polyNodes[0]] + localData[polyNodes[2]] + localData[polyNodes[3]] + localData[polyNodes[4]] - (isoValue * 4);
            if (cellVal < 0)
                numPoly += 2;
            else
                numPoly += 4;
            break;
        case 4:
            o = cI[i].i * leftRight + cI[i].j * upDown + cI[i].k;
            localData[0] = dataPtr[o + upDown + 1];
            localData[1] = dataPtr[o + upDown];
            localData[2] = dataPtr[o + upDown + leftRight];
            localData[3] = dataPtr[o + upDown + leftRight + 1];
            localData[4] = dataPtr[o + 1];
            localData[5] = dataPtr[o];
            localData[6] = dataPtr[o + leftRight];
            localData[7] = dataPtr[o + leftRight + 1];
            polyNodes = mCubesTbl[cI[i].bitmap].nodePairs;
            cellVal = localData[polyNodes[0]] + localData[polyNodes[2]] + localData[polyNodes[3]] + localData[polyNodes[4]] - (isoValue * 4);
            if (cellVal > 0)
                numPoly += 2;
            else
                numPoly += 4;
            break;
        case 5:
            numPoly += 3;
            break;
        case 6:
            numPoly += 2;
            break;
        case 7:
            numPoly += 2;
            break;
        case 8:
            numPoly += 3;
            break;
        case 9:
            numPoly += 4;
            break;
        case 10:
            numPoly += 3;
            break;
        case 11:
            numPoly += 4;
            break;
        case 12:
            numPoly += 4;
            break;
        case 13:
            numPoly += 4;
            break;
        case 14:
            numPoly += 4;
            break;
        case 15:
            numPoly += 4;
            break;
        }

        // coords
        j = cubeCutTable[cI[i].bitmap];
        k = cubeNumCoord[cI[i].bitmap];
        resortPtr = cI[i].nl;
        for (t = 0; t < 19; t++)
        {
            if (j & (1 << t))
            {
                a = mergeTable[t][0];
                b = mergeTable[t][1];
                if (*(resortPtr + a) > -1)
                {
                    if (cI[*(resortPtr + a)].vertOK)
                        k--;
                    else if (a != b && cI[*(resortPtr + a)].nl[b] > -1)
                    {
                        if (cI[cI[*(resortPtr + a)].nl[b]].vertOK)
                            k--;
                    }
                }
                else if (a != b && *(resortPtr + b) > -1)
                {
                    if (cI[*(resortPtr + b)].vertOK)
                        k--;
                    else if (cI[*(resortPtr + b)].nl[a] > -1)
                    {
                        if (cI[cI[*(resortPtr + b)].nl[a]].vertOK)
                            k--;
                    }
                }
            }
            // skip 16
            if (t == 15)
                t++;
        }

        // done
        numCoord += k;
        cI[i].vertOK = 1;
    }

    // triangles only
    numVert = numPoly * 3;

    // reset
    for (i = 0; i < numHandle; i++)
        cI[i].vertOK = 0;

    // info
    //cerr << "numPoly=" << numPoly << endl;
    //cerr << "numVert=" << numVert << endl;
    //cerr << "numCoord=" << numCoord << endl;

    //cerr << "Grid-Info:  min:  " << xMin << " " << yMin << " " << zMin << endl;
    //cerr << "            d:    " << dx << " " << dy << " " << dz << endl;

    // build output
    poly = new coDoPolygons(outName, numCoord, numVert, numPoly);
    poly->getAddresses(&xCoord, &yCoord, &zCoord, &vertList, &polyList);
    poly->addAttribute("vertexOrder", "2");

    // test
    for (i = 0; i < numVert; i++)
        vertList[i] = 0;

    numPoly = 0;
    numVert = 0;
    numCoord = 0;
    leftRight = numJ * numK;
    upDown = numK;
    for (i = 0; i < numHandle; i++)
    {
        j = cubeCutTable[cI[i].bitmap];

        // get data
        o = cI[i].i * leftRight + cI[i].j * upDown + cI[i].k;
        localData[0] = dataPtr[o + upDown + 1];
        localData[1] = dataPtr[o + upDown];
        localData[2] = dataPtr[o + upDown + leftRight];
        localData[3] = dataPtr[o + upDown + leftRight + 1];
        localData[4] = dataPtr[o + 1];
        localData[5] = dataPtr[o];
        localData[6] = dataPtr[o + leftRight];
        localData[7] = dataPtr[o + leftRight + 1];

        // and coords.
        localCoord[5][0] = xMin + ((float)cI[i].i) * dx;
        localCoord[5][1] = yMin + ((float)cI[i].j) * dy;
        localCoord[5][2] = zMin + ((float)cI[i].k) * dz;
        localCoord[0][0] = localCoord[5][0];
        localCoord[0][1] = localCoord[5][1] + dy;
        localCoord[0][2] = localCoord[5][2] + dz;
        localCoord[1][0] = localCoord[5][0];
        localCoord[1][1] = localCoord[5][1] + dy;
        localCoord[1][2] = localCoord[5][2];
        localCoord[2][0] = localCoord[5][0] + dx;
        localCoord[2][1] = localCoord[5][1] + dy;
        localCoord[2][2] = localCoord[5][2];
        localCoord[3][0] = localCoord[2][0];
        localCoord[3][1] = localCoord[2][1];
        localCoord[3][2] = localCoord[2][2] + dz;
        localCoord[4][0] = localCoord[5][0];
        localCoord[4][1] = localCoord[5][1];
        localCoord[4][2] = localCoord[5][2] + dz;
        localCoord[6][0] = localCoord[5][0] + dx;
        localCoord[6][1] = localCoord[5][1];
        localCoord[6][2] = localCoord[5][2];
        localCoord[7][0] = localCoord[6][0];
        localCoord[7][1] = localCoord[6][1];
        localCoord[7][2] = localCoord[6][2] + dz;

        // get/compute vertices
        resortPtr = cI[i].nl;
        for (t = 0; t < 12; t++)
        {
            cI[i].vl[t] = -1;
            if (j & (1 << t))
            {
                a = mergeTable[t][0];
                b = mergeTable[t][1];
                if (t != 16)
                {
                    if (*(resortPtr + a) > -1)
                    {
                        if (cI[*(resortPtr + a)].vertOK)
                            cI[i].vl[t] = cI[*(resortPtr + a)].vl[mergeTable[t][2]];
                        else if (a != b && cI[*(resortPtr + a)].nl[b] > -1)
                        {
                            if (cI[cI[*(resortPtr + a)].nl[b]].vertOK)
                                cI[i].vl[t] = cI[cI[*(resortPtr + a)].nl[b]].vl[mergeTable[t][4]];
                        }
                    }
                    else if (a != b && *(resortPtr + b) > -1)
                    {
                        if (cI[*(resortPtr + b)].vertOK)
                            cI[i].vl[t] = cI[*(resortPtr + b)].vl[mergeTable[t][3]];
                        else if (cI[*(resortPtr + b)].nl[a] > -1)
                        {
                            if (cI[cI[*(resortPtr + b)].nl[a]].vertOK)
                                cI[i].vl[t] = cI[cI[*(resortPtr + b)].nl[a]].vl[mergeTable[t][4]];
                        }
                    }
                }
                if (cI[i].vl[t] == -1)
                {
                    // compute
                    cI[i].vl[t] = numCoord;
                    float x1, y1, z1, d1;
                    float x2, y2, z2, d2;
                    float s;

                    // init
                    x1 = localCoord[nodesTable[t][0]][0];
                    y1 = localCoord[nodesTable[t][0]][1];
                    z1 = localCoord[nodesTable[t][0]][2];
                    d1 = localData[nodesTable[t][0]];
                    x2 = localCoord[nodesTable[t][1]][0];
                    y2 = localCoord[nodesTable[t][1]][1];
                    z2 = localCoord[nodesTable[t][1]][2];
                    d2 = localData[nodesTable[t][1]];

                    // interpolate (linear)
                    if (d2 == d1)
                    {
                        s = 0;
                    }
                    else
                        s = (isoValue - d1) / (d2 - d1);

                    xCoord[numCoord] = x1 + (x2 - x1) * s;
                    yCoord[numCoord] = y1 + (y2 - y1) * s;
                    zCoord[numCoord] = z1 + (z2 - z1) * s;

                    /* 
               float w2, w1;
               if( d2==d1 )
                  w2 = 1.0;
               else
               {
                  w2 = (d1-isoValue)/(d1-d2);
                  if( w2>1.0 )
                     w2 = 1.0;
                  else
                  if( w2<0.0 )
               w2 = 0.0;
               }
               w1 = 1.0-w2;
               xCoord[numCoord] = x1*w1 + x2*w2;
               yCoord[numCoord] = y1*w1 + y2*w2;
               zCoord[numCoord] = z1*w1 + z2*w2;
               */

                    numCoord++;
                }
            }
        }
        cI[i].vertOK = 1;

        // and triangulate
        polyNodes = mCubesTbl[cI[i].bitmap].nodePairs;

        //cerr << "  " << mCubesTbl[cI[i].bitmap].caseSel << endl;
        //cerr << "   " << (int)cI[i].bitmap << endl;

        switch (mCubesTbl[cI[i].bitmap].caseSel)
        {
        case 1:
        {
            int pN[3][2] = {
                { 0, 1 },
                { 0, 2 },
                { 0, 3 }
            };
            polyList[numPoly] = numVert;
            numPoly++;
            for (j = 0; j < 3; j++)
            {
                vertList[numVert] = cI[i].vl[edgeLookUp[polyNodes[pN[j][0]]][polyNodes[pN[j][1]]]];
                numVert++;
            }
            break;
        }
        case 2:
        {
            int pN[6][2] = {
                { 0, 1 },
                { 0, 2 },
                { 3, 4 },
                { 0, 1 },
                { 3, 4 },
                { 3, 5 }
            };
            polyList[numPoly] = numVert;
            polyList[numPoly + 1] = numVert + 3;
            numPoly += 2;
            for (j = 0; j < 6; j++)
            {
                vertList[numVert] = cI[i].vl[edgeLookUp[polyNodes[pN[j][0]]][polyNodes[pN[j][1]]]];
                numVert++;
            }
            break;
        }
        case 3:
            o = cI[i].i * leftRight + cI[i].j * upDown + cI[i].k;
            localData[0] = dataPtr[o + upDown + 1];
            localData[1] = dataPtr[o + upDown];
            localData[2] = dataPtr[o + upDown + leftRight];
            localData[3] = dataPtr[o + upDown + leftRight + 1];
            localData[4] = dataPtr[o + 1];
            localData[5] = dataPtr[o];
            localData[6] = dataPtr[o + leftRight];
            localData[7] = dataPtr[o + leftRight + 1];
            cellVal = localData[polyNodes[0]] + localData[polyNodes[2]] + localData[polyNodes[3]] + localData[polyNodes[4]] - (isoValue * 4);
            if (cellVal < 0)
            {
                polyList[numPoly] = numVert;
                polyList[numPoly + 1] = numVert + 3;
                numPoly += 2;
                int pN[6][2] = {
                    { 0, 1 },
                    { 0, 2 },
                    { 0, 3 },
                    { 4, 3 },
                    { 4, 2 },
                    { 4, 5 }
                };
                for (j = 0; j < 6; j++)
                {
                    vertList[numVert] = cI[i].vl[edgeLookUp[polyNodes[pN[j][0]]][polyNodes[pN[j][1]]]];
                    numVert++;
                }
            }
            else
            {
                polyList[numPoly] = numVert;
                polyList[numPoly + 1] = numVert + 3;
                polyList[numPoly + 2] = numVert + 6;
                polyList[numPoly + 3] = numVert + 9;
                numPoly += 4;
                int pN[12][2] = {
                    { 0, 1 },
                    { 0, 2 },
                    { 4, 2 },
                    { 0, 1 },
                    { 4, 2 },
                    { 4, 5 },
                    { 0, 3 },
                    { 0, 1 },
                    { 4, 5 },
                    { 0, 3 },
                    { 4, 5 },
                    { 4, 3 }
                };
                for (j = 0; j < 12; j++)
                {
                    vertList[numVert] = cI[i].vl[edgeLookUp[polyNodes[pN[j][0]]][polyNodes[pN[j][1]]]];
                    numVert++;
                }
            }
            break;
        case 4:
            o = cI[i].i * leftRight + cI[i].j * upDown + cI[i].k;
            localData[0] = dataPtr[o + upDown + 1];
            localData[1] = dataPtr[o + upDown];
            localData[2] = dataPtr[o + upDown + leftRight];
            localData[3] = dataPtr[o + upDown + leftRight + 1];
            localData[4] = dataPtr[o + 1];
            localData[5] = dataPtr[o];
            localData[6] = dataPtr[o + leftRight];
            localData[7] = dataPtr[o + leftRight + 1];
            /*
            cellVal = localData[polyNodes[0]] + localData[polyNodes[2]] + localData[polyNodes[3]] + localData[polyNodes[4]] - (isoValue*4);
            if( cellVal>0 )
            {
               polyList[numPoly] = numVert;
               polyList[numPoly+1] = numVert+3;
               numPoly+=2;
               int pN[6][2] = { {0, 1}, {0, 2}, {0, 3}, {4, 3}, {4, 2}, {4, 5} };
               for( j=0; j<6; j++ )
               {
                  vertList[numVert] = cI[i].vl[edgeLookUp[polyNodes[pN[j][0]]][polyNodes[pN[j][1]]]];
            numVert++;
            }
            }
            else*/
            {
                polyList[numPoly] = numVert;
                polyList[numPoly + 1] = numVert + 3;
                polyList[numPoly + 2] = numVert + 6;
                polyList[numPoly + 3] = numVert + 9;
                numPoly += 4;
                int pN[12][2] = {
                    { 0, 1 },
                    { 0, 2 },
                    { 4, 2 },
                    { 0, 1 },
                    { 4, 2 },
                    { 4, 5 },
                    { 0, 3 },
                    { 0, 1 },
                    { 4, 5 },
                    { 0, 3 },
                    { 4, 5 },
                    { 4, 3 }
                };
                for (j = 0; j < 12; j++)
                {
                    vertList[numVert] = cI[i].vl[edgeLookUp[polyNodes[pN[j][0]]][polyNodes[pN[j][1]]]];
                    numVert++;
                }
            }
            break;
        case 5:
        {
            //int pN[9][2] = { {0, 1}, {0, 2}, {3, 4}, {0, 1}, {3, 4}, {5, 6}, {0, 1}, {5, 6}, {5, 7} };
            int pN[9][2] = {
                { 0, 1 },
                { 0, 2 },
                { 3, 4 },
                { 0, 1 },
                { 3, 4 },
                { 5, 7 },
                { 5, 7 },
                { 3, 4 },
                { 5, 6 }
            };
            polyList[numPoly] = numVert;
            polyList[numPoly + 1] = numVert + 3;
            polyList[numPoly + 2] = numVert + 6;
            numPoly += 3;
            for (j = 0; j < 9; j++)
            {
                vertList[numVert] = cI[i].vl[edgeLookUp[polyNodes[pN[j][0]]][polyNodes[pN[j][1]]]];
                numVert++;
            }
            break;
        }
        case 6:
        {
            int pN[6][2] = {
                { 0, 1 },
                { 2, 3 },
                { 4, 5 },
                { 0, 1 },
                { 4, 5 },
                { 6, 7 }
            };
            polyList[numPoly] = numVert;
            polyList[numPoly + 1] = numVert + 3;
            numPoly += 2;
            for (j = 0; j < 6; j++)
            {
                vertList[numVert] = cI[i].vl[edgeLookUp[polyNodes[pN[j][0]]][polyNodes[pN[j][1]]]];
                numVert++;
            }
            break;
        }
        case 7:
        {
            int pN[6][2] = {
                { 0, 1 },
                { 0, 2 },
                { 0, 3 },
                { 4, 5 },
                { 4, 6 },
                { 4, 7 }
            };
            polyList[numPoly] = numVert;
            polyList[numPoly + 1] = numVert + 3;
            numPoly += 2;
            for (j = 0; j < 6; j++)
            {
                vertList[numVert] = cI[i].vl[edgeLookUp[polyNodes[pN[j][0]]][polyNodes[pN[j][1]]]];
                numVert++;
            }
            break;
        }
        case 8:
        {
            int pN[9][2] = {
                { 0, 1 },
                { 0, 2 },
                { 3, 4 },
                { 0, 1 },
                { 3, 4 },
                { 3, 5 },
                { 6, 7 },
                { 6, 8 },
                { 6, 9 }
            };
            polyList[numPoly] = numVert;
            polyList[numPoly + 1] = numVert + 3;
            polyList[numPoly + 2] = numVert + 6;
            numPoly += 3;
            for (j = 0; j < 9; j++)
            {
                vertList[numVert] = cI[i].vl[edgeLookUp[polyNodes[pN[j][0]]][polyNodes[pN[j][1]]]];

                /*
               if( cI[i].bitmap==138 )
               {
                  cerr << vertList[numVert] << "  " << pN[j][0] << "-" << pN[j][1] << endl;
                  cerr << "   -> " << polyNodes[pN[j][0]] << "-" << polyNodes[pN[j][1]];
                  cerr << "  -> " << edgeLookUp[polyNodes[pN[j][0]]][polyNodes[pN[j][1]]] << endl;
               }
               */

                numVert++;
            }
            break;
        }
        case 9:
        {
            //int pN[12][2] = { {0, 1}, {2, 3}, {2, 4}, {0, 1}, {2, 4}, {5, 4}, \ 
            //       {0, 1}, {5, 4}, {6, 7}, {0, 1}, {6, 7}, {6, 8} };
            int pN[12][2] = {
                { 0, 1 },
                { 2, 3 },
                { 6, 8 },
                { 2, 3 },
                { 2, 4 },
                { 6, 8 },
                { 2, 4 },
                { 5, 4 },
                { 6, 8 },
                { 5, 4 },
                { 6, 7 },
                { 6, 8 }
            };
            polyList[numPoly] = numVert;
            polyList[numPoly + 1] = numVert + 3;
            polyList[numPoly + 2] = numVert + 6;
            polyList[numPoly + 3] = numVert + 9;
            numPoly += 4;
            for (j = 0; j < 12; j++)
            {
                vertList[numVert] = cI[i].vl[edgeLookUp[polyNodes[pN[j][0]]][polyNodes[pN[j][1]]]];
                numVert++;
            }
            break;
        }
        case 10:
        {
            int pN[9][2] = {
                { 0, 1 },
                { 0, 2 },
                { 0, 3 },
                { 4, 5 },
                { 4, 6 },
                { 4, 7 },
                { 8, 9 },
                { 8, 10 },
                { 8, 11 }
            };
            polyList[numPoly] = numVert;
            polyList[numPoly + 1] = numVert + 3;
            polyList[numPoly + 2] = numVert + 6;
            numPoly += 3;
            for (j = 0; j < 9; j++)
            {
                vertList[numVert] = cI[i].vl[edgeLookUp[polyNodes[pN[j][0]]][polyNodes[pN[j][1]]]];
                numVert++;
            }
            break;
        }
        case 11:
        {
            int pN[12][2] = {
                { 0, 1 },
                { 0, 2 },
                { 3, 4 },
                { 0, 1 },
                { 3, 4 },
                { 3, 5 },
                { 0, 1 },
                { 3, 5 },
                { 6, 7 },
                { 0, 1 },
                { 6, 7 },
                { 6, 8 }
            };
            polyList[numPoly] = numVert;
            polyList[numPoly + 1] = numVert + 3;
            polyList[numPoly + 2] = numVert + 6;
            polyList[numPoly + 3] = numVert + 9;
            numPoly += 4;
            for (j = 0; j < 12; j++)
            {
                vertList[numVert] = cI[i].vl[edgeLookUp[polyNodes[pN[j][0]]][polyNodes[pN[j][1]]]];
                numVert++;
            }
            break;
        }
        case 12:
        {
            int pN[12][2] = {
                { 0, 1 },
                { 0, 2 },
                { 3, 4 },
                { 0, 1 },
                { 3, 4 },
                { 5, 4 },
                { 0, 1 },
                { 5, 4 },
                { 5, 6 },
                { 0, 1 },
                { 5, 6 },
                { 8, 7 }
            };
            polyList[numPoly] = numVert;
            polyList[numPoly + 1] = numVert + 3;
            polyList[numPoly + 2] = numVert + 6;
            polyList[numPoly + 3] = numVert + 9;
            numPoly += 4;
            for (j = 0; j < 12; j++)
            {
                vertList[numVert] = cI[i].vl[edgeLookUp[polyNodes[pN[j][0]]][polyNodes[pN[j][1]]]];
                numVert++;
            }
            break;
        }
        case 13:
        {
            int pN[12][2] = {
                { 0, 1 },
                { 0, 2 },
                { 0, 3 },
                { 4, 5 },
                { 4, 6 },
                { 7, 6 },
                { 4, 5 },
                { 7, 6 },
                { 7, 8 },
                { 4, 5 },
                { 7, 8 },
                { 9, 10 }
            };
            //int pN[12][2] = { {0, 1}, {0, 2}, {0, 3}, {4, 5}, {4, 6}, {10, 5}, \ 
            //       {10, 5}, {4, 6}, {9, 10}, {4, 6}, {7, 8}, {9, 10} };
            polyList[numPoly] = numVert;
            polyList[numPoly + 1] = numVert + 3;
            polyList[numPoly + 2] = numVert + 6;
            polyList[numPoly + 3] = numVert + 9;
            numPoly += 4;
            for (j = 0; j < 12; j++)
            {
                vertList[numVert] = cI[i].vl[edgeLookUp[polyNodes[pN[j][0]]][polyNodes[pN[j][1]]]];
                numVert++;
            }
            break;
        }
        case 14:
        {
            int pN[12][2] = {
                { 0, 1 },
                { 0, 2 },
                { 3, 4 },
                { 0, 1 },
                { 3, 4 },
                { 3, 5 },
                { 6, 7 },
                { 6, 8 },
                { 9, 10 },
                { 6, 7 },
                { 9, 10 },
                { 9, 11 }
            };
            polyList[numPoly] = numVert;
            polyList[numPoly + 1] = numVert + 3;
            polyList[numPoly + 2] = numVert + 6;
            polyList[numPoly + 3] = numVert + 9;
            numPoly += 4;
            for (j = 0; j < 12; j++)
            {
                vertList[numVert] = cI[i].vl[edgeLookUp[polyNodes[pN[j][0]]][polyNodes[pN[j][1]]]];
                numVert++;
            }
            break;
        }
        case 15:
        {
            int pN1[12][2] = {
                { 1, 0 },
                { 1, 5 },
                { 1, 2 },
                { 4, 5 },
                { 4, 0 },
                { 4, 7 },
                { 6, 2 },
                { 6, 5 },
                { 6, 7 },
                { 3, 0 },
                { 3, 2 },
                { 3, 7 }
            };
            int pN2[12][2] = {
                { 0, 1 },
                { 0, 3 },
                { 0, 4 },
                { 5, 1 },
                { 5, 4 },
                { 5, 6 },
                { 2, 1 },
                { 2, 6 },
                { 2, 3 },
                { 7, 4 },
                { 7, 3 },
                { 7, 6 }
            };
            polyList[numPoly] = numVert;
            polyList[numPoly + 1] = numVert + 3;
            polyList[numPoly + 2] = numVert + 6;
            polyList[numPoly + 3] = numVert + 9;
            numPoly += 4;
            if (polyNodes[0])
            {
                for (j = 0; j < 12; j++)
                {
                    vertList[numVert] = cI[i].vl[edgeLookUp[pN1[j][0]][pN1[j][1]]];
                    numVert++;
                }
            }
            else
            {
                for (j = 0; j < 12; j++)
                {
                    vertList[numVert] = cI[i].vl[edgeLookUp[pN2[j][0]][pN2[j][1]]];
                    numVert++;
                }
            }
            break;
        }
        }
    }

    //cerr << "----------------------" << endl;
    //cerr << "numPoly after: " << numPoly << endl;
    //cerr << "numVert after: " << numVert << endl;
    //cerr << "numCoord after: " << numCoord << endl;
    //cerr << endl;

    //for( i=0; i<numStrips; i++ )
    //   cerr << stripList[i] << endl;
    j = 0;
    for (i = 0; i < numVert; i++)
        if (vertList[i] < 0 || vertList[i] >= numVert)
        {
            //cerr << vertList[i] << "  ";
            if (!j)
            {
                cerr << (int)cI[0].bitmap << endl;
                j = 1;
            }
            vertList[i] = 0;
        }
    //cerr << endl;

    //for( i=0; i<numCoord; i++ )
    //   cerr << xCoord[i] << " " << yCoord[i] << " " << zCoord[i] << endl;

    // add some feedback-stuff
    if (poly)
    {
        char buf[1024];
        sprintf(buf, "I%s\n%s\n%s\n", Covise::get_module(), Covise::get_instance(), Covise::get_host());
        poly->addAttribute("FEEDBACK", buf);
    }

    // final clean up
    delete[] cI;

    //cerr << "done" << endl << endl;

    // done
    return (poly);
}

void IsoSurfaceS::computeMCubesTbl()
{
    int i, j;
    char bm, b1, b2, b3, b4, b5, b6, b7, b8;
    int te, cnc;
    // tct[6],
    //   cerr << "computing cube-table";

    for (i = 0; i < 256; i++)
    {
        // i = cube-bitmap
        bm = (char)i;
        b1 = bm & 1;
        b2 = bm & 2 ? 1 : 0;
        b3 = bm & 4 ? 1 : 0;
        b4 = bm & 8 ? 1 : 0;
        b5 = bm & 16 ? 1 : 0;
        b6 = bm & 32 ? 1 : 0;
        b7 = bm & 64 ? 1 : 0;
        b8 = bm & 128 ? 1 : 0;

        te = 0;

        // where will we have to compute vertices ?!
        if (b1 + b2 == 1)
            te |= 1;
        if (b1 + b5 == 1)
            te |= 1 << 8;
        if (b1 + b4 == 1)
            te |= 1 << 3;
        if (b2 + b3 == 1)
            te |= 1 << 1;
        if (b2 + b6 == 1)
            te |= 1 << 9;
        if (b3 + b7 == 1)
            te |= 1 << 10;
        if (b3 + b4 == 1)
            te |= 1 << 2;
        if (b4 + b8 == 1)
            te |= 1 << 11;
        if (b5 + b6 == 1)
            te |= 1 << 4;
        if (b5 + b8 == 1)
            te |= 1 << 7;
        if (b6 + b7 == 1)
            te |= 1 << 5;
        if (b7 + b8 == 1)
            te |= 1 << 6;

        cubeCutTable[i] = te;

        // cubeNumCoord ?!
        cnc = 0;
        for (j = 0; j < 12; j++)
        {
            if (te & (1 << j))
                cnc++;
        }
        cubeNumCoord[i] = cnc;

        //if( i==138 )
        //{
        //   cerr << endl << te << "  " << cnc << endl;
        //}
    }

    //   cerr << endl << "   done" << endl;

    // done
    return;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

coDistributedObject *IsoSurfaceS::issSkeletonClimbing(coDoUniformGrid *grid, coDoFloat *data, char *outName)
{
    // output vars
    coDoTriangleStrips *triStrips = NULL;
    //int numStrips, numVert, numCoord;
    //float *xCoord, *yCoord, *zCoord;
    //int *vertList, *stripList;

    // input vars
    float *dataPtr = NULL;
    int numI, numJ, numK;
    float xMin, yMin, zMin; //dx, dy, dz,
    float xMax, yMax, zMax;

    // own stuff
    char *nodes = NULL;
    int i; // ,j, k, t, o, a, b
    // int leftRight, upDown;
    float *floatPtr;
    char *nodesPtr; //*cellsPtr,
    //int *resortPtr;
    //typedef struct cellInfo_s {
    //     int i, j, k;  // position of cell
    //     char bitmap;
    //     int vl[12];   // vertices
    //     int nl[6];    // neighbor cells
    //     char vertOK;  // set if vl[] valid
    //     } cellInfo;
    //cellInfo *cI;
    //int *resortTable;
    //   char bm, b1, b2, b3, b4, b5, b6, b7, b8;
    //float localData[8], localCoord[8][3];

    // get input
    grid->getMinMax(&xMin, &xMax, &yMin, &yMax, &zMin, &zMax);
    grid->getGridSize(&numI, &numJ, &numK);
    //grid->getDelta( &dx, &dy, &dz );  // BUGGY !!!
    //dx = (xMax-xMin)/(numI-1);
    //dy = (yMax-yMin)/(numJ-1);
    //dz = (zMax-zMin)/(numK-1);

    data->getAddress(&dataPtr);
    //leftRight = numJ*numK;
    //upDown = numK;

    // first run, determine number of cells to handle
    //cells = new char[(numI-1)*(numJ-1)*(numK-1)];
    nodes = new char[numI * numJ * numK];
    //resortTable = new int[(numI-1)*(numJ-1)*(numK-1)];
    floatPtr = dataPtr;
    nodesPtr = nodes;
    for (i = 0; i < numI * numJ * numK; i++)
    {
        if (*floatPtr >= isoValue)
            *nodesPtr = 1;
        else
            *nodesPtr = 0;
        floatPtr++;
        nodesPtr++;
    }

    // done
    return (triStrips);
}
