/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <appl/ApplInterface.h>
#include "Grid.h"

#include <util/coviseCompat.h>
#include <do/coDoData.h>

#undef DEBUG

int twelveCount = 0;
int heavyCount = 0;

int flipCount = 0;
int fr_case = 0;
int rl_case = 0;
int tb_case = 0;

Grid::Grid(const coDistributedObject *gridIn, const coDistributedObject *dataIn)
    : gridObjIn((const coDoUnstructuredGrid *)gridIn)
{
    const coDoUnstructuredGrid *usgIn;
    const coDoFloat *sDataIn;
    const coDoVec3 *vDataIn;

    // get input grid
    usgIn = (const coDoUnstructuredGrid *)gridIn;
    usgIn->getGridSize(&numElem, &numConn, &numPoints);
    usgIn->getAddresses(&elemList, &connList, &xCoord, &yCoord, &zCoord);
    usgIn->getTypeList(&typeList);

    scalarFlag = 0;
    vectorFlag = 0;
    // and get the data
    const char *objType = dataIn->getType();
    if (!strcmp(objType, "USTSDT"))
    {
        // we have scalar data
        scalarFlag = 1;
        vectorFlag = 0;
        sDataIn = (const coDoFloat *)dataIn;
        if (sDataIn->getNumPoints() != numPoints)
        {
            scalarFlag = 0;
            uData = vData = wData = NULL;
        }
        else
        {
            sDataIn->getAddress(&uData);
            vData = wData = NULL;
        }
    }
    else if (!strcmp(objType, "USTVDT"))
    {
        // we have vector data
        scalarFlag = 0;
        vectorFlag = 1;
        vDataIn = (const coDoVec3 *)dataIn;
        if (vDataIn->getNumPoints() != numPoints)
        {
            vectorFlag = 0;
            uData = vData = wData = NULL;
        }
        else
            vDataIn->getAddresses(&uData, &vData, &wData);
    }
    else
    {
        // unsupported data
        scalarFlag = vectorFlag = 0;
        uData = vData = wData = NULL;
    }

    // other stuff
    gridObjName = NULL;
    dataObjName = NULL;
    neighborList = NULL;
    neighborIndexList = NULL;
    elementProcessed = NULL;
    regulateFlag = 0;

    // done
    return;
}

Grid::Grid(char *gridName, char *dataName)
{
    // allocate mem
    gridObjName = new char[strlen(gridName) + 1];
    dataObjName = new char[strlen(dataName) + 1];

    // keep the object-names
    strcpy(gridObjName, gridName);
    strcpy(dataObjName, dataName);

    // other stuff
    neighborList = NULL;
    neighborIndexList = NULL;
    elementProcessed = NULL;
    regulateFlag = 0;
    uData = vData = wData = NULL;

    // done
    return;
}

Grid::~Grid()
{
    // clean up
    if (gridObjName)
        delete[] gridObjName;
    if (dataObjName)
        delete[] dataObjName;
    if (elementProcessed)
        delete[] elementProcessed;

    // done
    return;
}

void Grid::computeNeighborList()
{
    // let covise do this job
    if (!neighborList || !neighborIndexList)
        gridObjIn->getNeighborList(&numNeighbors, &neighborList, &neighborIndexList);

    //
    // this neighbor-list is as follows:
    //   neighborIndexList[ coordNo ] -> neighborList
    // while the neighborList has numNeighbors entries
    //

    // done
    return;
}

coDistributedObject **Grid::tetrahedronize(Grid *oldGrid, int regulate)
{
    // output objects
    coDistributedObject **returnObject = NULL;
    /*
   coDoUnstructuredGrid *usgOut = NULL;
   coDoFloat *sDataOut = NULL;
   coDoVec3 *vDataOut = NULL;*/

    // user information
    int oldSize, newSize;

    // counters

    // initialize
    regulateFlag = regulate;
    //returnObject = new coDistributedObject*[2];
    //returnObject[0] = returnObject[1] = NULL;

    // compute size of old grid
    oldSize = oldGrid->numElem * 2 + oldGrid->numConn + oldGrid->numPoints * 3;

    if (regulateFlag)
    {
        // we should try to generate a regular grid
    }
    else
    {
        // just transform the grid without keeping track of regularity

        // and perform the transform
        returnObject = irregularTransformation(oldGrid);
    }

    // compute size of new grid
    newSize = numElem * 2 + numConn + numPoints * 3;

    // compute percentage and let the user know
    Covise::sendInfo("new grid has %6.2f%% the size of the original", (((float)newSize) / ((float)oldSize)) * 100.0);

    // sk : 11.04.2001
    // fprintf(stderr, "twelveCount is %d\n", twelveCount);
    // fprintf(stderr, "heavyCount is %d\n", heavyCount);
    // fprintf(stderr, "tb: %d   fr: %d   rl: %d   flip: %d   total: %d\n",
    // 		tb_case, fr_case, rl_case, flipCount, tb_case+fr_case+rl_case);

    // done
    return (returnObject);
}

int Grid::computeNewSize(Grid *grid)
{
    int i;
    int r = 0;

    // reset
    numElem = numConn = numPoints = 0;

    // this differs whether we have to regulate the grid or not
    if (regulateFlag)
    {
    }
    else
    {
        // this is quite simple
        for (i = 0; i < grid->numElem; i++)
        {
            switch (grid->typeList[i])
            {
            case TYPE_HEXAGON:
                numElem += 6;
                numConn += 24; // 6 * 4
                break;
            case TYPE_PRISM:
                numElem += 3;
                numConn += 12; // 3 * 4
                break;
            case TYPE_PYRAMID:
                numElem += 2;
                numConn += 8; // 2 * 4
                break;
            case TYPE_TETRAHEDER:
                numElem++;
                numConn += 4; // 1 * 4
                break;
            default:
                // ignore this element
                break;
            }
        }
    }

    // done
    return (r);
}

///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

//int twelveCount = 0;

int Grid::haveToSplitPrism(int s0, int s1, int s2)
{
    // find any valid case and return 0 if so

    if (s0 == s1)
        return (0);

    if (s0 == s2)
        return (0);

    if (s1 != s2)
        return (0);

    // damn
    return (1);
}

void Grid::splitPrism(int el, Grid *grid, int doFlag)
{
    int c0, c1, c2, c3, c4, c5;
    int s0, s1, s2;
    int o;

    int sixFlag = 0;

    int tetra[6][4];

    int t;
    int i;

    // get data
    o = grid->elemList[el];
    c0 = grid->connList[o];
    c1 = grid->connList[o + 1];
    c2 = grid->connList[o + 2];
    c3 = grid->connList[o + 3];
    c4 = grid->connList[o + 4];
    c5 = grid->connList[o + 5];

    s0 = elementSide[0][el];
    s1 = elementSide[1][el];
    s2 = elementSide[2][el];

    // do we have to perform 6-decomposition ?
    sixFlag = haveToSplitPrism(s0, s1, s2);

    //   fprintf(stderr, "splitPrism %d: sixFlag=%d\n", el, sixFlag);

    if (!doFlag)
    {
        // just analyse the case
        if (sixFlag)
        {
            numPoints++;
            numConn += 6 * 4;
            numElem += 6;
        }
        else
        {
            numConn += 3 * 4;
            numElem += 3;
        }

        // done
        return;
    }

    // some further initialization
    // this is for easier programming
    for (t = 0; t < (sixFlag ? 6 : 3); t++)
        for (i = 0; i < 4; i++)
            tetra[t][i] = numConn + (t * 4) + i;

    // anyway we will get 3 or 6 tetrahedrons
    for (i = 0; i < (sixFlag ? 6 : 3); i++)
    {
        typeList[numElem + i] = TYPE_TETRAHEDER;
        elemList[numElem + i] = numConn + (i * 4);
    }

    // allways compose into three tetrahedra here, for testing purposes

    /*
   if( s0==(char)-2 || s1==(char)-2 || s2==(char)-2 )
   {
      fprintf(stderr, "ERROR: one of the sides is -2, but all should be 1 or 2 !\n");
   }

   if( s0==(char)-2 )
      s0 = 1;
   if( s1==(char)-2 )
      s1 = 1;
   if( s2==(char)-2 )
   s2 = 1;*/

    // perform the splitting
    if (sixFlag)
    {
        // argh
        float xS, yS, zS;
        int S;

        //      fprintf(stderr, "performing 6-split of prism %d: %d %d %d\n", el, s0, s1, s2);

        xS = grid->xCoord[c0] + grid->xCoord[c1] + grid->xCoord[c2] + grid->xCoord[c3] + grid->xCoord[c4] + grid->xCoord[c5];
        yS = grid->yCoord[c0] + grid->yCoord[c1] + grid->yCoord[c2] + grid->yCoord[c3] + grid->yCoord[c4] + grid->yCoord[c5];
        zS = grid->zCoord[c0] + grid->zCoord[c1] + grid->zCoord[c2] + grid->zCoord[c3] + grid->zCoord[c4] + grid->zCoord[c5];

        xS /= 6.0;
        yS /= 6.0;
        zS /= 6.0;

        xCoord[numPoints] = xS;
        yCoord[numPoints] = yS;
        zCoord[numPoints] = zS;
        S = numPoints;

        // now the data
        if (grid->isInsideGetData(c0, c2, c3, c4, xS, yS, zS, uData[numPoints], vData[numPoints], wData[numPoints]))
        {
            // first one matches
        }
        else if (grid->isInsideGetData(c0, c4, c1, c2, xS, yS, zS, uData[numPoints], vData[numPoints], wData[numPoints]))
        {
            // second one matches
        }
        else if (grid->isInsideGetData(c5, c2, c4, c3, xS, yS, zS, uData[numPoints], vData[numPoints], wData[numPoints]))
        {
            // third one matches
        }
        else
        {
            fprintf(stderr, "NOT GOOD: splitPrism\n");
        }

        // finally decompose the element
        if (s0 == 1)
        {
            connList[tetra[0][0]] = c0;
            connList[tetra[0][1]] = c2;
            connList[tetra[0][2]] = c5;
            connList[tetra[0][3]] = S;

            connList[tetra[1][0]] = c0;
            connList[tetra[1][1]] = c5;
            connList[tetra[1][2]] = c3;
            connList[tetra[1][3]] = S;
        }
        else
        {
            connList[tetra[0][0]] = c0;
            connList[tetra[0][1]] = c2;
            connList[tetra[0][2]] = c3;
            connList[tetra[0][3]] = S;

            connList[tetra[1][0]] = c3;
            connList[tetra[1][1]] = c2;
            connList[tetra[1][2]] = c5;
            connList[tetra[1][3]] = S;
        }

        if (s1 == 1)
        {
            connList[tetra[2][0]] = c1;
            connList[tetra[2][1]] = c5;
            connList[tetra[2][2]] = c2;
            connList[tetra[2][3]] = S;

            connList[tetra[3][0]] = c1;
            connList[tetra[3][1]] = c4;
            connList[tetra[3][2]] = c5;
            connList[tetra[3][3]] = S;
        }
        else
        {
            connList[tetra[2][0]] = c1;
            connList[tetra[2][1]] = c4;
            connList[tetra[2][2]] = c2;
            connList[tetra[2][3]] = S;

            connList[tetra[3][0]] = c2;
            connList[tetra[3][1]] = c4;
            connList[tetra[3][2]] = c5;
            connList[tetra[3][3]] = S;
        }

        if (s2 == 1)
        {
            connList[tetra[4][0]] = c0;
            connList[tetra[4][1]] = c3;
            connList[tetra[4][2]] = c4;
            connList[tetra[4][3]] = S;

            connList[tetra[5][0]] = c0;
            connList[tetra[5][1]] = c4;
            connList[tetra[5][2]] = c1;
            connList[tetra[5][3]] = S;
        }
        else
        {
            connList[tetra[4][0]] = c0;
            connList[tetra[4][1]] = c3;
            connList[tetra[4][2]] = c1;
            connList[tetra[4][3]] = S;

            connList[tetra[5][0]] = c1;
            connList[tetra[5][1]] = c3;
            connList[tetra[5][2]] = c4;
            connList[tetra[5][3]] = S;
        }
    }
    else
    {
        // do it

        //      fprintf(stderr, "performing 3-split of prism %d: %d %d %d\n", el, s0, s1, s2);

        if (s0 == 1)
        {
            if (s2 == 1)
            {
                connList[tetra[0][0]] = c0;
                connList[tetra[0][1]] = c5;
                connList[tetra[0][2]] = c3;
                connList[tetra[0][3]] = c4;

                if (s1 == 1)
                {
                    connList[tetra[1][0]] = c1;
                    connList[tetra[1][1]] = c4;
                    connList[tetra[1][2]] = c5;
                    connList[tetra[1][3]] = c0;

                    connList[tetra[2][0]] = c1;
                    connList[tetra[2][1]] = c5;
                    connList[tetra[2][2]] = c2;
                    connList[tetra[2][3]] = c0;
                }
                else
                {
                    connList[tetra[1][0]] = c1;
                    connList[tetra[1][1]] = c4;
                    connList[tetra[1][2]] = c2;
                    connList[tetra[1][3]] = c0;

                    connList[tetra[2][0]] = c2;
                    connList[tetra[2][1]] = c4;
                    connList[tetra[2][2]] = c5;
                    connList[tetra[2][3]] = c0;
                }
            }
            else
            {
                connList[tetra[0][0]] = c0;
                connList[tetra[0][1]] = c5;
                connList[tetra[0][2]] = c3;
                connList[tetra[0][3]] = c1;

                // s1 has to be one here
                connList[tetra[1][0]] = c0;
                connList[tetra[1][1]] = c2;
                connList[tetra[1][2]] = c5;
                connList[tetra[1][3]] = c1;

                connList[tetra[2][0]] = c5;
                connList[tetra[2][1]] = c1;
                connList[tetra[2][2]] = c4;
                connList[tetra[2][3]] = c3;
            }

            /*
            connList[tetra[0][0]] = c;
            connList[tetra[0][1]] = c;
            connList[tetra[0][2]] = c;
            connList[tetra[0][3]] = c;

            connList[tetra[0][0]] = c;
            connList[tetra[0][1]] = c;
            connList[tetra[0][2]] = c;
            connList[tetra[0][3]] = c;

         connList[tetra[0][0]] = c;
         connList[tetra[0][1]] = c;
         connList[tetra[0][2]] = c;
         connList[tetra[0][3]] = c;
         */
        }
        else
        {

            if (s2 == 2)
            {
                connList[tetra[0][0]] = c0;
                connList[tetra[0][1]] = c2;
                connList[tetra[0][2]] = c3;
                connList[tetra[0][3]] = c1;

                if (s1 == 1)
                {
                    connList[tetra[1][0]] = c2;
                    connList[tetra[1][1]] = c1;
                    connList[tetra[1][2]] = c5;
                    connList[tetra[1][3]] = c3;

                    connList[tetra[2][0]] = c1;
                    connList[tetra[2][1]] = c4;
                    connList[tetra[2][2]] = c5;
                    connList[tetra[2][3]] = c3;
                }
                else
                {
                    connList[tetra[1][0]] = c1;
                    connList[tetra[1][1]] = c4;
                    connList[tetra[1][2]] = c2;
                    connList[tetra[1][3]] = c3;

                    connList[tetra[2][0]] = c2;
                    connList[tetra[2][1]] = c4;
                    connList[tetra[2][2]] = c5;
                    connList[tetra[2][3]] = c3;
                }
            }
            else
            {
                connList[tetra[0][0]] = c0;
                connList[tetra[0][1]] = c2;
                connList[tetra[0][2]] = c3;
                connList[tetra[0][3]] = c4;

                // s1 has to be two here
                connList[tetra[1][0]] = c2;
                connList[tetra[1][1]] = c5;
                connList[tetra[1][2]] = c3;
                connList[tetra[1][3]] = c4;

                connList[tetra[2][0]] = c2;
                connList[tetra[2][1]] = c1;
                connList[tetra[2][2]] = c4;
                connList[tetra[2][3]] = c0;
            }
        }
    }

    // done
    if (sixFlag)
    {
        numPoints++;
        numConn += 6 * 4;
        numElem += 6;
    }
    else
    {
        numConn += 3 * 4;
        numElem += 3;
    }

    // done
    return;
}

void Grid::splitHexa(int el, Grid *grid, int doFlag)
{
    int c0, c1, c2, c3, c4, c5, c6, c7;
    int s1, s3, s4, s5;
    int tetra[12][4];

    int t;
    int i;
    int heavyFlag = 0;
    int twelveFlag = 0;

    // init
    c0 = c1 = c2 = c3 = c4 = c5 = c6 = c7 = 0; // we love to get "no warning messages"
    s1 = s3 = s4 = s5 = 0;
    createIdealHexagon(el, grid, c0, c1, c2, c3, c4, c5, c6, c7, s1, s3, s4, s5);
    if (doFlag)
    {
        // this is for easier programming
        for (t = 0; t < 12; t++)
            for (i = 0; i < 4; i++)
                tetra[t][i] = numConn + (t * 4) + i;

        // anyway we will get 6 tetrahedrons
        for (i = 0; i < 6; i++)
        {
            typeList[numElem + i] = TYPE_TETRAHEDER;
            elemList[numElem + i] = numConn + (i * 4);
        }
    }

    // check how we might do it (i left, t right: 0 no matter, 1 or 2)
    i = 0;
    if (s3 == 1 && s5 == 1)
        i = 1;
    if (s3 == 2 && s5 == 2)
        i = 2;

    t = 0;
    if (s4 == 1 && s1 == 2)
        t = 1;
    if (s4 == 2 && s1 == 1)
        t = 2;

    if (t != i)
    {
        // one of them should be free to decide, otherwise we can't build
        // a regular grid (BANG)
        if (t && i)
            heavyFlag = 1;
    }

    if (t == 0 && i == 0)
        t = i = 1;
    else if (t == 0)
        t = i;
    else if (i == 0)
        i = t;

    if (s1 == -1 && s3 == -1 && s4 == -1 && s5 == -1)
    {
        heavyFlag = 0;
        twelveFlag = 1;
    }

    // maybe we should only compute the size
    if (!doFlag)
    {
        if (twelveFlag)
        {
            numElem += 12;
            numConn += 12 * 4;
            numPoints++;
        }
        else if (heavyFlag)
        {
            numElem += 11;
            numConn += 11 * 4;
            numPoints++;
        }
        else
        {
            numElem += 6;
            numConn += 6 * 4;
        }

        // done
        return;
    }

    // handle left prism

    //   if( el==1 || el==36 )
    //   {
    //      fprintf(stderr, "el: %d\n", el);
    //      fprintf(stderr, "twelveFlag: %d   heavyFlag: %d\n", twelveFlag, heavyFlag);
    //      fprintf(stderr, "   s1: %d   s3: %d   s4: %d   s5:%d\n", s1, s3, s4, s5);

    //   }

    if (twelveFlag)
    {
        // decompose into twelve tetrahedra
        // compute centroid and data at that point
        float xS, yS, zS;
        int S;

        xS = grid->xCoord[c0] + grid->xCoord[c1] + grid->xCoord[c2] + grid->xCoord[c3] + grid->xCoord[c4] + grid->xCoord[c5] + grid->xCoord[c6] + grid->xCoord[c7];
        yS = grid->yCoord[c0] + grid->yCoord[c1] + grid->yCoord[c2] + grid->yCoord[c3] + grid->yCoord[c4] + grid->yCoord[c5] + grid->yCoord[c6] + grid->yCoord[c7];
        zS = grid->zCoord[c0] + grid->zCoord[c1] + grid->zCoord[c2] + grid->zCoord[c3] + grid->zCoord[c4] + grid->zCoord[c5] + grid->zCoord[c6] + grid->zCoord[c7];

        xS /= 8.0;
        yS /= 8.0;
        zS /= 8.0;

        xCoord[numPoints] = xS;
        yCoord[numPoints] = yS;
        zCoord[numPoints] = zS;
        S = numPoints;

        // now the data
        if (grid->isInsideGetData(c0, c4, c5, c6, xS, yS, zS, uData[numPoints], vData[numPoints], wData[numPoints]))
        {
            // first one matches
        }
        else if (grid->isInsideGetData(c0, c5, c1, c6, xS, yS, zS, uData[numPoints], vData[numPoints], wData[numPoints]))
        {
            // second one matches
        }
        else if (grid->isInsideGetData(c2, c1, c6, c0, xS, yS, zS, uData[numPoints], vData[numPoints], wData[numPoints]))
        {
            // third one matches
        }
        else if (grid->isInsideGetData(c4, c0, c7, c6, xS, yS, zS, uData[numPoints], vData[numPoints], wData[numPoints]))
        {
            // fourth one matches
        }
        else if (grid->isInsideGetData(c0, c3, c7, c2, xS, yS, zS, uData[numPoints], vData[numPoints], wData[numPoints]))
        {
            // fifth one matches
        }
        else if (grid->isInsideGetData(c7, c2, c6, c0, xS, yS, zS, uData[numPoints], vData[numPoints], wData[numPoints]))
        {
            // sixth one matches
        }

        // finally decompose the element
        if (elementSide[0][el] == 1)
        {
            connList[tetra[0][0]] = c0;
            connList[tetra[0][1]] = c1;
            connList[tetra[0][2]] = c2;
            connList[tetra[0][3]] = S;

            connList[tetra[1][0]] = c0;
            connList[tetra[1][1]] = c2;
            connList[tetra[1][2]] = c3;
            connList[tetra[1][3]] = S;
        }
        else
        {
            connList[tetra[0][0]] = c0;
            connList[tetra[0][1]] = c1;
            connList[tetra[0][2]] = c3;
            connList[tetra[0][3]] = S;

            connList[tetra[1][0]] = c1;
            connList[tetra[1][1]] = c2;
            connList[tetra[1][2]] = c3;
            connList[tetra[1][3]] = S;
        }

        if (elementSide[1][el] == 1)
        {
            connList[tetra[2][0]] = c3;
            connList[tetra[2][1]] = c2;
            connList[tetra[2][2]] = c7;
            connList[tetra[2][3]] = S;

            connList[tetra[3][0]] = c7;
            connList[tetra[3][1]] = c2;
            connList[tetra[3][2]] = c6;
            connList[tetra[3][3]] = S;
        }
        else
        {
            connList[tetra[2][0]] = c3;
            connList[tetra[2][1]] = c2;
            connList[tetra[2][2]] = c6;
            connList[tetra[2][3]] = S;

            connList[tetra[3][0]] = c3;
            connList[tetra[3][1]] = c6;
            connList[tetra[3][2]] = c7;
            connList[tetra[3][3]] = S;
        }

        if (elementSide[2][el] == 1)
        {
            connList[tetra[4][0]] = c4;
            connList[tetra[4][1]] = c7;
            connList[tetra[4][2]] = c6;
            connList[tetra[4][3]] = S;

            connList[tetra[5][0]] = c4;
            connList[tetra[5][1]] = c6;
            connList[tetra[5][2]] = c5;
            connList[tetra[5][3]] = S;
        }
        else
        {
            connList[tetra[4][0]] = c4;
            connList[tetra[4][1]] = c7;
            connList[tetra[4][2]] = c5;
            connList[tetra[4][3]] = S;

            connList[tetra[5][0]] = c7;
            connList[tetra[5][1]] = c6;
            connList[tetra[5][2]] = c5;
            connList[tetra[5][3]] = S;
        }

        if (elementSide[3][el] == 1)
        {
            connList[tetra[6][0]] = c0;
            connList[tetra[6][1]] = c5;
            connList[tetra[6][2]] = c1;
            connList[tetra[6][3]] = S;

            connList[tetra[7][0]] = c0;
            connList[tetra[7][1]] = c4;
            connList[tetra[7][2]] = c5;
            connList[tetra[7][3]] = S;
        }
        else
        {
            connList[tetra[6][0]] = c0;
            connList[tetra[6][1]] = c4;
            connList[tetra[6][2]] = c1;
            connList[tetra[6][3]] = S;

            connList[tetra[7][0]] = c1;
            connList[tetra[7][1]] = c4;
            connList[tetra[7][2]] = c5;
            connList[tetra[7][3]] = S;
        }

        if (elementSide[4][el] == 1)
        {
            connList[tetra[8][0]] = c0;
            connList[tetra[8][1]] = c3;
            connList[tetra[8][2]] = c7;
            connList[tetra[8][3]] = S;

            connList[tetra[9][0]] = c0;
            connList[tetra[9][1]] = c7;
            connList[tetra[9][2]] = c4;
            connList[tetra[9][3]] = S;
        }
        else
        {
            connList[tetra[8][0]] = c0;
            connList[tetra[8][1]] = c3;
            connList[tetra[8][2]] = c4;
            connList[tetra[8][3]] = S;

            connList[tetra[9][0]] = c3;
            connList[tetra[9][1]] = c7;
            connList[tetra[9][2]] = c4;
            connList[tetra[9][3]] = S;
        }

        if (elementSide[5][el] == 1)
        {
            connList[tetra[10][0]] = c2;
            connList[tetra[10][1]] = c1;
            connList[tetra[10][2]] = c6;
            connList[tetra[10][3]] = S;

            connList[tetra[11][0]] = c1;
            connList[tetra[11][1]] = c5;
            connList[tetra[11][2]] = c6;
            connList[tetra[11][3]] = S;
        }
        else
        {
            connList[tetra[10][0]] = c2;
            connList[tetra[10][1]] = c1;
            connList[tetra[10][2]] = c5;
            connList[tetra[10][3]] = S;

            connList[tetra[11][0]] = c2;
            connList[tetra[11][1]] = c5;
            connList[tetra[11][2]] = c6;
            connList[tetra[11][3]] = S;
        }

        //uData[c0] = vData[c0] = wData[c0] = 1000.0;
        //uData[c1] = vData[c1] = wData[c1] = 2000.0;
        //uData[c2] = vData[c2] = wData[c2] = 3000.0;
        //uData[c3] = vData[c3] = wData[c3] = 4000.0;
        //uData[c4] = vData[c4] = wData[c4] = 5000.0;
        //uData[c5] = vData[c5] = wData[c5] = 6000.0;
        //uData[c6] = vData[c6] = wData[c6] = 7000.0;
        //uData[c7] = vData[c7] = wData[c7] = 8000.0;
        //uData[numPoints] = vData[numPoints] = wData[numPoints] = 9000.0;

        // don't forgett to add the tetrahedra
        for (i = 6; i < 12; i++)
        {
            typeList[numElem + i] = TYPE_TETRAHEDER;
            elemList[numElem + i] = numConn + (i * 4);
        }
    }
    else
        // if heavyFlag is set, then decompose the left prism so that it fits the
        // right's triangulation
        if (heavyFlag)
    {
        // compute centroid and data at that point
        float xS, yS, zS;
        int S;

        xS = grid->xCoord[c0] + grid->xCoord[c1] + grid->xCoord[c2] + grid->xCoord[c4] + grid->xCoord[c5] + grid->xCoord[c6];
        yS = grid->yCoord[c0] + grid->yCoord[c1] + grid->yCoord[c2] + grid->yCoord[c4] + grid->yCoord[c5] + grid->yCoord[c6];
        zS = grid->zCoord[c0] + grid->zCoord[c1] + grid->zCoord[c2] + grid->zCoord[c4] + grid->zCoord[c5] + grid->zCoord[c6];

        xS /= 6.0;
        yS /= 6.0;
        zS /= 6.0;

        xCoord[numPoints] = xS;
        yCoord[numPoints] = yS;
        zCoord[numPoints] = zS;
        S = numPoints;

        // now the data
        // we use 3- decomposition and check in which component
        // the centroid is situated, then calculate the data in that
        // point with the well-known "volume"-method

        if (grid->isInsideGetData(c0, c4, c5, c6, xS, yS, zS, uData[numPoints], vData[numPoints], wData[numPoints]))
        {
            // first one matches
        }
        else if (grid->isInsideGetData(c0, c5, c1, c6, xS, yS, zS, uData[numPoints], vData[numPoints], wData[numPoints]))
        {
            // second one matches
        }
        else if (grid->isInsideGetData(c2, c1, c6, c0, xS, yS, zS, uData[numPoints], vData[numPoints], wData[numPoints]))
        {
            // third one matches
        }

        // decompose the prism
        i = t;

        if (s3 == 1)
        {
            connList[tetra[0][0]] = c0;
            connList[tetra[0][1]] = c4;
            connList[tetra[0][2]] = c5;
            connList[tetra[0][3]] = S;

            connList[tetra[1][0]] = c0;
            connList[tetra[1][1]] = c5;
            connList[tetra[1][2]] = c1;
            connList[tetra[1][3]] = S;
        }
        else
        {
            connList[tetra[0][0]] = c0;
            connList[tetra[0][1]] = c4;
            connList[tetra[0][2]] = c1;
            connList[tetra[0][3]] = S;

            connList[tetra[1][0]] = c4;
            connList[tetra[1][1]] = c5;
            connList[tetra[1][2]] = c1;
            connList[tetra[1][3]] = S;
        }

        if (s5 == 1)
        {
            connList[tetra[2][0]] = c2;
            connList[tetra[2][1]] = c1;
            connList[tetra[2][2]] = c6;
            connList[tetra[2][3]] = S;

            connList[tetra[6][0]] = c1;
            connList[tetra[6][1]] = c5;
            connList[tetra[6][2]] = c6;
            connList[tetra[6][3]] = S;
        }
        else
        {
            connList[tetra[2][0]] = c2;
            connList[tetra[2][1]] = c1;
            connList[tetra[2][2]] = c5;
            connList[tetra[2][3]] = S;

            connList[tetra[6][0]] = c2;
            connList[tetra[6][1]] = c5;
            connList[tetra[6][2]] = c6;
            connList[tetra[6][3]] = S;
        }

        if (i == 1)
        {
            connList[tetra[7][0]] = c0;
            connList[tetra[7][1]] = c2;
            connList[tetra[7][2]] = c6;
            connList[tetra[7][3]] = S;

            connList[tetra[8][0]] = c0;
            connList[tetra[8][1]] = c6;
            connList[tetra[8][2]] = c4;
            connList[tetra[8][3]] = S;
        }
        else
        {
            connList[tetra[7][0]] = c0;
            connList[tetra[7][1]] = c2;
            connList[tetra[7][2]] = c4;
            connList[tetra[7][3]] = S;

            connList[tetra[8][0]] = c4;
            connList[tetra[8][1]] = c2;
            connList[tetra[8][2]] = c6;
            connList[tetra[8][3]] = S;
        }

        // don't forget the "caps"
        connList[tetra[9][0]] = c0;
        connList[tetra[9][1]] = c1;
        connList[tetra[9][2]] = c2;
        connList[tetra[9][3]] = S;

        connList[tetra[10][0]] = c4;
        connList[tetra[10][1]] = c6;
        connList[tetra[10][2]] = c5;
        connList[tetra[10][3]] = S;

        //uData[c0] = vData[c0] = wData[c0] = 1000.0;
        //uData[c1] = vData[c1] = wData[c1] = 2000.0;
        //uData[c2] = vData[c2] = wData[c2] = 3000.0;
        //uData[c3] = vData[c3] = wData[c3] = 4000.0;
        //uData[c4] = vData[c4] = wData[c4] = 5000.0;
        //uData[c5] = vData[c5] = wData[c5] = 6000.0;
        //uData[c6] = vData[c6] = wData[c6] = 7000.0;
        //uData[c7] = vData[c7] = wData[c7] = 8000.0;
        //uData[numPoints] = vData[numPoints] = wData[numPoints] = 9000.0;

        // remember to add enough more elements
        for (i = 6; i < 11; i++)
        {
            typeList[numElem + i] = TYPE_TETRAHEDER;
            elemList[numElem + i] = numConn + (i * 4);
        }

        // tramdalam
    }
    else if (s3 == 1 && s5 == 1)
    {
        // i has to be 1 in this case
        connList[tetra[0][0]] = c0;
        connList[tetra[0][1]] = c4;
        connList[tetra[0][2]] = c5;
        connList[tetra[0][3]] = c6;

        connList[tetra[1][0]] = c0;
        connList[tetra[1][1]] = c5;
        connList[tetra[1][2]] = c1;
        connList[tetra[1][3]] = c6;

        connList[tetra[2][0]] = c2;
        connList[tetra[2][1]] = c1;
        connList[tetra[2][2]] = c6;
        connList[tetra[2][3]] = c0;

        if (i != 1)
        {
            fprintf(stderr, "BUG BUG BUG !!!\n");
        }

        //uData[c0] = uData[c1] = uData[c2] = uData[c4] = uData[c5] = uData[c6] = 1000.0;
    }
    else if (s3 == 1 && s5 == 2)
    {
        if (i == 1)
        {
            connList[tetra[0][0]] = c0;
            connList[tetra[0][1]] = c4;
            connList[tetra[0][2]] = c5;
            connList[tetra[0][3]] = c6;

            connList[tetra[1][0]] = c6;
            connList[tetra[1][1]] = c2;
            connList[tetra[1][2]] = c5;
            connList[tetra[1][3]] = c0;

            connList[tetra[2][0]] = c1;
            connList[tetra[2][1]] = c0;
            connList[tetra[2][2]] = c5;
            connList[tetra[2][3]] = c2;
        }
        else
        {
            connList[tetra[0][0]] = c0;
            connList[tetra[0][1]] = c4;
            connList[tetra[0][2]] = c5;
            connList[tetra[0][3]] = c2;

            connList[tetra[1][0]] = c1;
            connList[tetra[1][1]] = c0;
            connList[tetra[1][2]] = c5;
            connList[tetra[1][3]] = c2;

            connList[tetra[2][0]] = c6;
            connList[tetra[2][1]] = c2;
            connList[tetra[2][2]] = c5;
            connList[tetra[2][3]] = c4;
        }
    }
    else if (s3 == 2 && s5 == 1)
    {
        if (i == 1)
        {
            connList[tetra[0][0]] = c2;
            connList[tetra[0][1]] = c1;
            connList[tetra[0][2]] = c6;
            connList[tetra[0][3]] = c0;

            connList[tetra[1][0]] = c5;
            connList[tetra[1][1]] = c6;
            connList[tetra[1][2]] = c1;
            connList[tetra[1][3]] = c4;

            connList[tetra[2][0]] = c0;
            connList[tetra[2][1]] = c4;
            connList[tetra[2][2]] = c1;
            connList[tetra[2][3]] = c6;
        }
        else
        {
            connList[tetra[0][0]] = c0;
            connList[tetra[0][1]] = c4;
            connList[tetra[0][2]] = c1;
            connList[tetra[0][3]] = c2;

            connList[tetra[1][0]] = c5;
            connList[tetra[1][1]] = c1;
            connList[tetra[1][2]] = c4;
            connList[tetra[1][3]] = c6;

            connList[tetra[2][0]] = c2;
            connList[tetra[2][1]] = c1;
            connList[tetra[2][2]] = c6;
            connList[tetra[2][3]] = c4;
        }
    }
    else if (s3 == 2 && s5 == 2)
    {
        // i has to be 2 in this case
        connList[tetra[0][0]] = c0;
        connList[tetra[0][1]] = c4;
        connList[tetra[0][2]] = c1;
        connList[tetra[0][3]] = c2;

        connList[tetra[1][0]] = c5;
        connList[tetra[1][1]] = c2;
        connList[tetra[1][2]] = c1;
        connList[tetra[1][3]] = c4;

        connList[tetra[2][0]] = c6;
        connList[tetra[2][1]] = c2;
        connList[tetra[2][2]] = c5;
        connList[tetra[2][3]] = c4;
    }

    if (!twelveFlag)
    {
        // handle right prism
        if (s4 == 1 && s1 == 1)
        {

            //uData[c0] = uData[c4] = uData[c3] = uData[c2] = uData[6] = uData[c7] = 2000.0;

            if (t == 1)
            {
                connList[tetra[3][0]] = c4;
                connList[tetra[3][1]] = c0;
                connList[tetra[3][2]] = c7;
                connList[tetra[3][3]] = c6;

                connList[tetra[4][0]] = c0;
                connList[tetra[4][1]] = c3;
                connList[tetra[4][2]] = c7;
                connList[tetra[4][3]] = c2;

                connList[tetra[5][0]] = c7;
                connList[tetra[5][1]] = c2;
                connList[tetra[5][2]] = c6;
                connList[tetra[5][3]] = c0;
            }
            else
            {
                connList[tetra[3][0]] = c7;
                connList[tetra[3][1]] = c2;
                connList[tetra[3][2]] = c6;
                connList[tetra[3][3]] = c4;

                connList[tetra[4][0]] = c4;
                connList[tetra[4][1]] = c0;
                connList[tetra[4][2]] = c7;
                connList[tetra[4][3]] = c2;

                connList[tetra[5][0]] = c0;
                connList[tetra[5][1]] = c3;
                connList[tetra[5][2]] = c7;
                connList[tetra[5][3]] = c2;
            }
        }
        else if (s4 == 1 && s1 == 2)
        {
            // t has to be 1 in this case

            if (t != 1)
            {
                fprintf(stderr, "tjtja und noch ein Bug\n");
            }

            /*
         if( uData )
            uData[c0] = uData[c1] = uData[c2] = uData[c3] = uData[c4] = uData[c5] = uData[c6] = uData[c7] = 1.0;
         */
            //uData[c4] = uData[c0] = uData[c7] = uData[c6] = uData[3] = uData[c2] = 3000.0;

            connList[tetra[3][0]] = c4;
            connList[tetra[3][1]] = c0;
            connList[tetra[3][2]] = c7;
            connList[tetra[3][3]] = c6;

            connList[tetra[4][0]] = c0;
            connList[tetra[4][1]] = c3;
            connList[tetra[4][2]] = c7;
            connList[tetra[4][3]] = c6;

            connList[tetra[5][0]] = c3;
            connList[tetra[5][1]] = c2;
            connList[tetra[5][2]] = c6;
            connList[tetra[5][3]] = c0;
        }
        else if (s4 == 2 && s1 == 1)
        {
            // t has to be 2 in this case

            //uData[c0] = uData[c3] = uData[c4] = uData[c2] = uData[c7] = uData[c6] = 4000.0;

            connList[tetra[3][0]] = c0;
            connList[tetra[3][1]] = c3;
            connList[tetra[3][2]] = c4;
            connList[tetra[3][3]] = c2;

            connList[tetra[4][0]] = c4;
            connList[tetra[4][1]] = c3;
            connList[tetra[4][2]] = c7;
            connList[tetra[4][3]] = c2;

            connList[tetra[5][0]] = c2;
            connList[tetra[5][1]] = c6;
            connList[tetra[5][2]] = c7;
            connList[tetra[5][3]] = c4;
        }
        else if (s4 == 2 && s1 == 2)
        {

            //uData[c0] = uData[c3] = uData[c4] = uData[

            if (t == 1)
            {
                connList[tetra[3][0]] = c0;
                connList[tetra[3][1]] = c3;
                connList[tetra[3][2]] = c4;
                connList[tetra[3][3]] = c6;

                connList[tetra[4][0]] = c4;
                connList[tetra[4][1]] = c3;
                connList[tetra[4][2]] = c7;
                connList[tetra[4][3]] = c6;

                connList[tetra[5][0]] = c3;
                connList[tetra[5][1]] = c2;
                connList[tetra[5][2]] = c6;
                connList[tetra[5][3]] = c0;
            }
            else
            {
                connList[tetra[3][0]] = c0;
                connList[tetra[3][1]] = c3;
                connList[tetra[3][2]] = c4;
                connList[tetra[3][3]] = c2;

                connList[tetra[4][0]] = c4;
                connList[tetra[4][1]] = c3;
                connList[tetra[4][2]] = c7;
                connList[tetra[4][3]] = c6;

                connList[tetra[5][0]] = c3;
                connList[tetra[5][1]] = c2;
                connList[tetra[5][2]] = c6;
                connList[tetra[5][3]] = c4;
            }
        }
    }

    // done

    /*
   if( numElem > 251700 && numElem < 251800 )
   {
      fprintf(stderr, "numElem: %d   twelveFlag: %d  heavyFlag: %d  %d %d %d %d\n", \ 
            numElem, twelveFlag, heavyFlag, s1, s3, s4, s5);
      fprintf(stderr, "build from Hexaheder-Element %d\n", el);

   }
   */

    if (twelveFlag)
    {
        numElem += 12;
        numConn += 12 * 4;
        numPoints++;

        twelveCount++;
    }
    else if (heavyFlag)
    {
        numElem += 11;
        numConn += 11 * 4;
        numPoints++;

        heavyCount++;
    }
    else
    {
        numElem += 6;
        numConn += 24;
    }
    return;
}

void Grid::splitPyra(int el, Grid *grid)
{

    int c0, c1, c2, c3, c4;
    int o;

    typeList[numElem] = TYPE_TETRAHEDER;
    typeList[numElem + 1] = TYPE_TETRAHEDER;
    elemList[numElem] = numConn;
    elemList[numElem + 1] = numConn + 4;

    o = grid->elemList[el];
    c0 = grid->connList[o];
    c1 = grid->connList[o + 1];
    c2 = grid->connList[o + 2];
    c3 = grid->connList[o + 3];
    c4 = grid->connList[o + 4];

    if (elementSide[0][el] == 1)
    {
        connList[numConn] = c0;
        connList[numConn + 1] = c1;
        connList[numConn + 2] = c2;
        connList[numConn + 3] = c4;

        connList[numConn + 4] = c0;
        connList[numConn + 5] = c2;
        connList[numConn + 6] = c3;
        connList[numConn + 7] = c4;
    }
    else
    {
        connList[numConn] = c1;
        connList[numConn + 1] = c2;
        connList[numConn + 2] = c3;
        connList[numConn + 3] = c4;

        connList[numConn + 4] = c0;
        connList[numConn + 5] = c1;
        connList[numConn + 6] = c3;
        connList[numConn + 7] = c4;
    }

    numConn += 8;
    numElem += 2;

    // done
    return;
}

void Grid::createIdealHexagon(int el, Grid *grid, int &c0, int &c1, int &c2, int &c3, int &c4, int &c5, int &c6, int &c7, int &s1, int &s3, int &s4, int &s5)
{
    int n = grid->elemList[el];
    int flip = 0;

    //   if( el==1 || el==36 )
    //   {
    //      fprintf(stderr, "el: %d\n", el);
    //      fprintf(stderr, "tb: %d  rl: %d   fr: %d   flip: %d\n", tb_case, rl_case, fr_case, flipCount);

    //   }

    // find first side that matches the ODC
    if (elementSide[0][el] == elementSide[2][el])
    {
        // top/bottom
        flip = (elementSide[0][el] == 2);
        // that's the way we want it, so leave it unchanged
        c0 = grid->connList[n];
        c1 = grid->connList[n + 1];
        c2 = grid->connList[n + 2];
        c3 = grid->connList[n + 3];
        c4 = grid->connList[n + 4];
        c5 = grid->connList[n + 5];
        c6 = grid->connList[n + 6];
        c7 = grid->connList[n + 7];
        s1 = elementSide[1][el];
        s3 = elementSide[3][el];
        s4 = elementSide[4][el];
        s5 = elementSide[5][el];

        /*
      if( uData )
         uData[c0] = uData[c1] = uData[c2] = uData[c3] = uData[c4] = uData[c5] = uData[c6] = uData[c7] = 2.0;
      */

        tb_case++;
    }
    else if ((elementSide[1][el] == 1 && elementSide[3][el] == 2) || (elementSide[1][el] == 2 && elementSide[3][el] == 1))
    {
        // right/left
        c0 = grid->connList[n + 3];
        c1 = grid->connList[n + 2];
        c2 = grid->connList[n + 6];
        c3 = grid->connList[n + 7];
        c4 = grid->connList[n];
        c5 = grid->connList[n + 1];
        c6 = grid->connList[n + 5];
        c7 = grid->connList[n + 4];
        s1 = elementSide[2][el];
        s3 = (elementSide[0][el] == 1) ? 2 : 1;
        s4 = (elementSide[4][el] == 1) ? 2 : 1;
        s5 = (elementSide[5][el] == 1) ? 2 : 1;

        flip = (elementSide[1][el] == 1);

        /*
      if( uData )
      {
         //fprintf(stderr, "/");
         uData[c0] = uData[c1] = uData[c2] = uData[c3] = uData[c4] = uData[c5] = uData[c6] = uData[c7] = 1.0;
      }
      */

        rl_case++;
    }
    else if (elementSide[4][el] == elementSide[5][el])
    {
        // front/rear
        c0 = grid->connList[n + 1];
        c1 = grid->connList[n + 5];
        c2 = grid->connList[n + 6];
        c3 = grid->connList[n + 2];
        c4 = grid->connList[n];
        c5 = grid->connList[n + 4];
        c6 = grid->connList[n + 7];
        c7 = grid->connList[n + 3];
        s1 = (elementSide[1][el] == 1) ? 2 : 1;
        s3 = (elementSide[3][el] == 1) ? 2 : 1;
        s4 = (elementSide[0][el] == 1) ? 2 : 1;
        s5 = (elementSide[2][el] == 1) ? 2 : 1;

        flip = (elementSide[4][el] == 2);

        /*
      if( uData )
         uData[c0] = uData[c1] = uData[c2] = uData[c3] = uData[c4] = uData[c5] = uData[c6] = uData[c7] = 2.0;
      */

        fr_case++;
    }
    else
    {
        //fprintf(stderr, "createIdealHexagon failed: ODC not fullfilled - either the grid or the algorithm is shit\n");
        //fprintf(stderr, "                           workaraund: decompose it into 12 tetrahedrons (not implemented)\n");
        c0 = grid->connList[n];
        c1 = grid->connList[n + 1];
        c2 = grid->connList[n + 2];
        c3 = grid->connList[n + 3];
        c4 = grid->connList[n + 4];
        c5 = grid->connList[n + 5];
        c6 = grid->connList[n + 6];
        c7 = grid->connList[n + 7];
        s1 = -1;
        s3 = -1;
        s4 = -1;
        s5 = -1;

        /*
      if( uData )
         uData[c0] = uData[c1] = uData[c2] = uData[c3] = uData[c4] = uData[c5] = uData[c6] = uData[c7] = 2.0;
      */
    }

    // we may have to "flip" the hexagon (rotate it 90degrees about vertical axis)
    if (flip)
    {
        // change sides
        n = s4;
        s4 = (s1 == 1) ? 2 : 1;
        s1 = s5;
        s5 = s3;
        s3 = (n == 1) ? 2 : 1;

        // and vertices
        // top
        n = c0;
        c0 = c3;
        c3 = c2;
        c2 = c1;
        c1 = n;

        // bottom
        n = c4;
        c4 = c7;
        c7 = c6;
        c6 = c5;
        c5 = n;

        /*

      if( uData )
         uData[c0] = uData[c1] = uData[c2] = uData[c3] = uData[c4] = uData[c5] = uData[c6] = uData[c7] = uData[c7]+10.0;
      */

        flipCount++;
    }

    //   if( el==1 || el==36 )
    //   {
    //      fprintf(stderr, "el: %d\n", el);
    //      fprintf(stderr, "tb: %d  rl: %d   fr: %d   flip: %d\n", tb_case, rl_case, fr_case, flipCount);

    //   }

    // done
    return;
}

void Grid::handlePrismSide(int el, int s, int c0, int c1, int c2, int c3, Grid *grid)
{

    int n, ns, nc0;

    if (grid->elementSide[s][el] == (char)-2)
    {
        // we have to do what our neighbor did

        n = haveSameNeighbor(el, c0, c1, c2, c3);
        if (n == -1)
        {
            fprintf(stderr, "expected to find a neighbor, but there is none !\n");
        }

        ns = findSharedSide(n, c0, c1, c2, c3);
        nc0 = getStart(n, ns);

        //      fprintf(stderr, "do same as neighbor: %d %d %d %d\n", n, ns, nc0, (int)grid->elementSide[ns][n]);

        // see if the neighbor-side has been split
        if (grid->elementSide[ns][n] == 1)
        {
            // it has been split
            if (nc0 == c0 || nc0 == c2)
                grid->elementSide[s][el] = 1;
            else
                grid->elementSide[s][el] = 2;
        }
        else if (grid->elementSide[ns][n] == 2)
        {
            // it has been split
            if (nc0 == c0 || nc0 == c2)
                grid->elementSide[s][el] = 2;
            else
                grid->elementSide[s][el] = 1;
        }
        else
        {
            // the neighbor-side has not yet been split
            if (grid->elementProcessed[n])
            {
                // it has allready been processed, so we have to decide
                // on that one and do the same at the neighbor-side

                // this is the same as "we can do what we like"
                if (s == 0)
                {
                    if (grid->elementSide[1][el] == 1 || grid->elementSide[1][el] == 2)
                        grid->elementSide[s][el] = grid->elementSide[1][el];
                    else if (grid->elementSide[2][el] == 1 || grid->elementSide[2][el] == 2)
                        grid->elementSide[s][el] = grid->elementSide[2][el];
                    else
                        grid->elementSide[s][el] = 1;
                }
                else if (s == 1)
                {
                    if (grid->elementSide[0][el] == 1 || grid->elementSide[0][el] == 2)
                        grid->elementSide[s][el] = grid->elementSide[0][el];
                    else if (grid->elementSide[2][el] == 1)
                        grid->elementSide[s][el] = 2;
                    else
                        //if( grid->elementSide[2][el]==2 )
                        //   grid->elementSide[s][el] = 1;
                        //else
                        grid->elementSide[s][el] = 1;
                }
                else if (s == 2)
                {
                    if (grid->elementSide[0][el] == 1 || grid->elementSide[0][el] == 2)
                        grid->elementSide[s][el] = grid->elementSide[0][el];
                    else if (grid->elementSide[1][el] == 1)
                        grid->elementSide[s][el] = 2;
                    else
                        //if( grid->elementSide[1][el]==2 )
                        //   grid->elementSide[s][el] = grid->elementSide[2][el];
                        //else
                        grid->elementSide[s][el] = 1;
                }

                // and do the same at the neighbor

                if (nc0 == c0 || nc0 == c2)
                    grid->elementSide[ns][n] = grid->elementSide[s][el];
                else
                    grid->elementSide[ns][n] = (grid->elementSide[s][el] == 1) ? 2 : 1;

                /*
                 if( grid->elementSide[s][el]==1 )
                 {
                    // it has been split
               if( nc0==c0 || nc0==c2 )
                  grid->elementSide[ns][n] = 1;
               else
                  grid->elementSide[ns][n] = 2;
                 }
                 else
                 if( grid->elementSide[s][el]==2 )
            {
            // it has been split
            if( nc0==c0 || nc0==c2 )
            grid->elementSide[ns][n] = 2;
            else
            grid->elementSide[ns][n] = 1;
            }
            */
            }
            else
            {
                // leave the decission to the neighbor, this might be best

                //fprintf(stderr, "%d leaving decision on side %d to neighbor %d\n", el,
            }
        }
    }
    else if (grid->elementSide[s][el] == (char)-1)
    {
        // we can do what we like
        // try to do it the best way, so we don't have to
        // split into 6 tetrahedra

        if (s == 0)
        {
            if (grid->elementSide[1][el] == 1 || grid->elementSide[1][el] == 2)
                grid->elementSide[s][el] = grid->elementSide[1][el];
            else if (grid->elementSide[2][el] == 1 || grid->elementSide[2][el] == 2)
                grid->elementSide[s][el] = grid->elementSide[2][el];
            else
                grid->elementSide[s][el] = 1;
        }
        else if (s == 1)
        {
            if (grid->elementSide[0][el] == 1 || grid->elementSide[0][el] == 2)
                grid->elementSide[s][el] = grid->elementSide[0][el];
            else if (grid->elementSide[2][el] == 1)
                grid->elementSide[s][el] = 2;
            else
                //if( grid->elementSide[2][el]==2 )
                //   grid->elementSide[s][el] = 1;
                //else
                grid->elementSide[s][el] = 1;
        }
        else if (s == 2)
        {
            if (grid->elementSide[0][el] == 1 || grid->elementSide[0][el] == 2)
                grid->elementSide[s][el] = grid->elementSide[0][el];
            else if (grid->elementSide[1][el] == 1)
                grid->elementSide[s][el] = 2;
            else
                //if( grid->elementSide[1][el]==2 )
                //   grid->elementSide[s][el] = grid->elementSide[2][el];
                //else
                grid->elementSide[s][el] = 1;
        }
    }

    // done
    return;
}

void Grid::handleHexaSide(int el, int s, int c0, int c1, int c2, int c3, int os, int sFlag, Grid *grid)
{
    int n, ns, nc0;

    //
    // this function should handle the given side s of the given hexahedron el
    // and take care on the ODC
    //   if( (el==34 || el==66) )//&& s==4 )
    //   {
    //      fprintf(stderr, "hHS: el %d   side %d     %d %d %d %d\n", el, s, c0, c1, c2, c3);
    //   }

    //   if( (el==1 && s==1) || (el==36 && s==1) )
    //   {
    //      fprintf(stderr, "el: %d    elementSide: %d\n", el, (signed short int)grid->elementSide[s][el]);

    //   }

    if (grid->elementSide[s][el] == (char)-2)
    {
        // we have to do what our neighbor did
        // so get him

        // somethings wrong here...hmmm

        n = haveSameNeighbor(el, c0, c1, c2, c3);
        if (n == -1)
        {
            fprintf(stderr, "expected to find a neighbor, but there is none !\n");
        }

        ns = findSharedSide(n, c0, c1, c2, c3);
        nc0 = getStart(n, ns);

        // see if the neighbor-side has been split
        if (grid->elementSide[ns][n] == 1)
        {
            // it has been split
            if (nc0 == c0 || nc0 == c2)
                grid->elementSide[s][el] = 1;
            else
                grid->elementSide[s][el] = 2;
        }
        else if (grid->elementSide[ns][n] == 2)
        {
            // it has been split
            if (nc0 == c0 || nc0 == c2)
                grid->elementSide[s][el] = 2;
            else
                grid->elementSide[s][el] = 1;
        }
        else
        {
            // the neighbor-side has not yet been split
            if (grid->elementProcessed[n])
            {
                // the neighbor relies on us to split the side right now

                // what did the opposite side do
                if (grid->elementSide[os][el] == 2)
                {
                    // it is split, so do the same
                    if (sFlag)
                        grid->elementSide[s][el] = (grid->elementSide[os][el] == 2) ? 1 : 2;
                    else
                        grid->elementSide[s][el] = grid->elementSide[os][el];
                }
                else if (grid->elementSide[os][el] == 1)
                {
                    // it is split, so do the same
                    if (sFlag)
                        grid->elementSide[s][el] = (grid->elementSide[os][el] == 2) ? 1 : 2;
                    else
                        grid->elementSide[s][el] = grid->elementSide[os][el];
                }
                else
                {
                    // the opposite side is not yet split, so we can do what we like
                    grid->elementSide[s][el] = 1;
                }

                // don't forgett the fuckin' neighbor

                if (nc0 == c0 || nc0 == c2)
                    //if( nc0==c0 )
                    grid->elementSide[ns][n] = grid->elementSide[s][el];
                else
                    grid->elementSide[ns][n] = (grid->elementSide[s][el] == 2) ? 1 : 2;
            }
            else
            {
                // we rely on the neighbor to do the work for us later on
                // when it's his turn
            }
        }
    }
    else if (grid->elementSide[s][el] == (char)-1)
    {
        // we can do whatever we like, we don't have to mess with no neighbor

        // what did the opposite side do
        if (grid->elementSide[os][el] == 2)
        {
            // it is split, so do the same
            if (sFlag)
                grid->elementSide[s][el] = (grid->elementSide[os][el] == 2) ? 1 : 2;
            else
                grid->elementSide[s][el] = grid->elementSide[os][el];
        }
        else if (grid->elementSide[os][el] == 1)
        {
            // it is split, so do the same
            if (sFlag)
                grid->elementSide[s][el] = (grid->elementSide[os][el] == 2) ? 1 : 2;
            else
                grid->elementSide[s][el] = grid->elementSide[os][el];
        }
        else
        {
            // the opposite side is not yet split, so if it has to take care
            // of its neighbor we don't do anything
            if (grid->elementSide[os][el] == (char)-2)
            {
                // do nothing, we will be split when processing the opposite side
            }
        }
    }
    else
    {
        // this side has allready been processed, so don't change anything
        // otherwise we could only make things worse
    }

    // done
    return;
}

///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

coDistributedObject **Grid::irregularTransformation(Grid *grid)
{
    int i;
    int el, s, n;
    int cl0, cl1, cl2, cl3, cl4, cl5, cl6, cl7;
    int nc0;

    coDistributedObject **returnObject;
    coDoUnstructuredGrid *usgOut = NULL;
    coDoFloat *sDataOut = NULL;
    coDoVec3 *vDataOut = NULL;

    returnObject = new coDistributedObject *[2];
    returnObject[0] = returnObject[1] = NULL;

    if (grid->uData == NULL)
    {
        Covise::sendError("Only Vector data supported right now");
        return (returnObject);
    }
    // reset
    for (i = 0; i < 6; i++)
    {
        elementSide[i] = new char[grid->numElem];
        for (el = 0; el < grid->numElem; el++)
            elementSide[i][el] = 0;
    }
    elementProcessed = new char[grid->numElem];
    for (el = 0; el < grid->numElem; el++)
        elementProcessed[el] = 0;
    numElem = numConn = 0;
    grid->computeNeighborList();

    //////
    ////// PREPROCESSING  -  1st STEP
    //////

    // work through all elements
    for (i = 0; i < grid->numElem; i++)
    {
        // get offset from source grid
        el = grid->elemList[i];

        switch (grid->typeList[i])
        {
        case TYPE_HEXAGON:
            // we have to find the best decomposition
            cl0 = grid->connList[el];
            cl1 = grid->connList[el + 1];
            cl2 = grid->connList[el + 2];
            cl3 = grid->connList[el + 3];
            cl4 = grid->connList[el + 4];
            cl5 = grid->connList[el + 5];
            cl6 = grid->connList[el + 6];
            cl7 = grid->connList[el + 7];

            // therfore we analyze each side for its own
            // (only if it has not allready been analyzed)
            if (!elementSide[0][i])
                elementSide[0][i] = (char)grid->analyzeQuad(this, i, cl0, cl1, cl2, cl3);
            if (!elementSide[1][i])
                elementSide[1][i] = (char)grid->analyzeQuad(this, i, cl2, cl6, cl7, cl3);
            if (!elementSide[2][i])
                elementSide[2][i] = (char)grid->analyzeQuad(this, i, cl4, cl7, cl6, cl5);
            if (!elementSide[3][i])
                elementSide[3][i] = (char)grid->analyzeQuad(this, i, cl0, cl4, cl5, cl1);
            if (!elementSide[4][i])
                elementSide[4][i] = (char)grid->analyzeQuad(this, i, cl0, cl3, cl4, cl7);
            if (!elementSide[5][i])
                elementSide[5][i] = (char)grid->analyzeQuad(this, i, cl1, cl5, cl6, cl2);

            // done
            break;
        case TYPE_PYRAMID:
            cl0 = grid->connList[el];
            cl1 = grid->connList[el + 1];
            cl2 = grid->connList[el + 2];
            cl3 = grid->connList[el + 3];

            fprintf(stderr, "y");

            if (!elementSide[0][i])
                elementSide[0][i] = (char)grid->analyzeQuad(this, i, cl0, cl1, cl2, cl3);

            break;
        case TYPE_PRISM:
            cl0 = grid->connList[el];
            cl1 = grid->connList[el + 1];
            cl2 = grid->connList[el + 2];
            cl3 = grid->connList[el + 3];
            cl4 = grid->connList[el + 4];
            cl5 = grid->connList[el + 5];

            //fprintf(stderr, "p");

            if (!elementSide[0][i])
                elementSide[0][i] = (char)grid->analyzeQuad(this, i, cl0, cl2, cl5, cl3);

            if (!elementSide[1][i])
                elementSide[1][i] = (char)grid->analyzeQuad(this, i, cl1, cl4, cl5, cl2);

            if (!elementSide[2][i])
                elementSide[2][i] = (char)grid->analyzeQuad(this, i, cl0, cl3, cl4, cl1);

            break;
        default:
            // ignore this element
            break;
        }
    }

    //////
    ////// PREPROCESSING  -  2nd STEP
    //////

    // at the same time compute size of output
    numElem = 0;
    numPoints = grid->numPoints;
    numConn = 0;

    // work through all elements
    for (i = 0; i < grid->numElem; i++)
    {
        // get offset
        el = grid->elemList[i];

        // let's play
        switch (grid->typeList[i])
        {
        case TYPE_HEXAGON:
            // rooky

            // work through all sides
            cl0 = grid->connList[el];
            cl1 = grid->connList[el + 1];
            cl2 = grid->connList[el + 2];
            cl3 = grid->connList[el + 3];
            cl4 = grid->connList[el + 4];
            cl5 = grid->connList[el + 5];
            cl6 = grid->connList[el + 6];
            cl7 = grid->connList[el + 7];

            // top
            grid->handleHexaSide(i, 0, cl0, cl1, cl2, cl3, 2, 0, this);

            // right
            grid->handleHexaSide(i, 1, cl2, cl6, cl7, cl3, 3, 1, this);

            // bottom
            grid->handleHexaSide(i, 2, cl4, cl7, cl6, cl5, 0, 0, this);

            // left
            grid->handleHexaSide(i, 3, cl0, cl4, cl5, cl1, 1, 1, this);

            // front
            // was 4 7 damn !
            grid->handleHexaSide(i, 4, cl0, cl3, cl7, cl4, 5, 0, this);

            // rear
            grid->handleHexaSide(i, 5, cl1, cl5, cl6, cl2, 4, 0, this);

            // done
            elementProcessed[i] = 1;
            break;
        case TYPE_PRISM:
            // intermediate
            cl0 = grid->connList[el];
            cl1 = grid->connList[el + 1];
            cl2 = grid->connList[el + 2];
            cl3 = grid->connList[el + 3];
            cl4 = grid->connList[el + 4];
            cl5 = grid->connList[el + 5];

            // front
            grid->handlePrismSide(i, 0, cl0, cl2, cl5, cl3, this);

            // right
            grid->handlePrismSide(i, 1, cl1, cl4, cl5, cl2, this);

            // left
            grid->handlePrismSide(i, 2, cl0, cl3, cl4, cl1, this);

            //	    fprintf(stderr, "intermediate mode not implemented (see source for details) !\n");

            // done
            elementProcessed[i] = 1;
            break;
        case TYPE_PYRAMID:
            // easy
            if (elementSide[0][i] == (char)-2)
            {
                cl0 = grid->connList[el];
                cl1 = grid->connList[el + 1];
                cl2 = grid->connList[el + 2];
                cl3 = grid->connList[el + 3];

                // we have to do the same as our neighbor
                n = grid->haveSameNeighbor(i, cl0, cl1, cl2, cl3);

                // see if he has allready been processed
                if (elementProcessed[n])
                {
                    // yes it has been
                    s = grid->findSharedSide(n, cl0, cl1, cl2, cl3);
                    nc0 = grid->getStart(n, s);

                    if (elementSide[s][n] == (char)-2)
                    {
                        // we can do what we want as long as we keep the neighbor
                        // up-to-date
                        elementSide[0][i] = 1;

                        // the neighbor may be "phase shifted", so check this out
                        if (nc0 == cl0 || nc0 == cl2)
                            elementSide[s][n] = 1;
                        else
                            // shifted
                            elementSide[s][n] = 2;
                    }
                    else
                    {
                        // we have to do what the neighbor did

                        // the neighbor may be "phase shifted", so check this out
                        if (nc0 == cl0 || nc0 == cl2)
                            elementSide[0][i] = elementSide[s][n];
                        else
                        {
                            // shifted
                            elementSide[0][i] = (elementSide[s][n] == 1) ? 2 : 1;
                        }

                        if (elementSide[s][n] <= 0)
                            fprintf(stderr, "BUG !\n");
                    }
                }
                else
                {
                    // leave the decission to the neighbor
                    // so just do nothing
                }
            }
            else if (elementSide[0][i] == (char)-1)
            {
                // do what we want
                elementSide[0][i] = 1;
            }

            // we will build two tetrehadrons
            numElem += 2;
            numConn += 8;

            // done
            elementProcessed[i] = 1;
            break;
        case TYPE_TETRAHEDER:
            numElem++;
            numConn += 4;
            break;
        default:
            // ignore this element
            break;
        }
    }

    // don't forgett the hexahedrons
    for (i = 0; i < grid->numElem; i++)
    {
        if (grid->typeList[i] == TYPE_HEXAGON)
            splitHexa(i, grid, 0);
        else if (grid->typeList[i] == TYPE_PRISM)
            splitPrism(i, grid, 0);
    }

    //////
    ////// final step
    //////

    // create output
    usgOut = new coDoUnstructuredGrid(gridObjName, numElem, numConn, numPoints, 1);
    usgOut->getAddresses(&elemList, &connList, &xCoord, &yCoord, &zCoord);
    usgOut->getTypeList(&typeList);
    returnObject[0] = usgOut;

    // and the data
    if (grid->scalarFlag)
    {
        // scalar data
        sDataOut = new coDoFloat(dataObjName, numPoints);
        sDataOut->getAddress(&uData);
        vData = wData = NULL;
        returnObject[1] = sDataOut;
        scalarFlag = 1;
        vectorFlag = 0;
    }
    else if (grid->vectorFlag)
    {
        // vector data
        vDataOut = new coDoVec3(dataObjName, numPoints);
        vDataOut->getAddresses(&uData, &vData, &wData);
        returnObject[1] = vDataOut;
        scalarFlag = 0;
        vectorFlag = 1;

        for (i = 0; i < numPoints; i++)
        {
            uData[i] = vData[i] = wData[i] = 0.0;
        }
    }
    else
    {
        // no data / invalid data (!?!)
        scalarFlag = vectorFlag = 0;
        uData = vData = wData = NULL;
    }

    // on with the show
    numElem = numConn = 0;
    numPoints = grid->numPoints;

    // this is where we finally decompose the elements into tetrahedrons
    for (i = 0; i < grid->numElem; i++)
    {
        el = grid->elemList[i];

        switch (grid->typeList[i])
        {
        case TYPE_HEXAGON:
            splitHexa(i, grid);
            break;
        case TYPE_PYRAMID:
            splitPyra(i, grid);
            break;
        case TYPE_PRISM:
            splitPrism(i, grid);
            break;
        case TYPE_TETRAHEDER:
            // simply keep the tetrahedron
            typeList[numElem] = TYPE_TETRAHEDER;
            elemList[numElem] = numConn;
            connList[numConn] = grid->connList[el];
            connList[numConn + 1] = grid->connList[el + 1];
            connList[numConn + 2] = grid->connList[el + 2];
            connList[numConn + 3] = grid->connList[el + 3];
            numElem++;
            numConn += 4;
            break;
        }
    }

    // copy coordinates and data
    if (vectorFlag)
    {
        for (i = 0; i < grid->numPoints; i++)
        {
            xCoord[i] = grid->xCoord[i];
            yCoord[i] = grid->yCoord[i];
            zCoord[i] = grid->zCoord[i];
            uData[i] = grid->uData[i];
            vData[i] = grid->vData[i];
            wData[i] = grid->wData[i];
        }
    }
    else if (scalarFlag)
    {
        for (i = 0; i < grid->numPoints; i++)
        {
            xCoord[i] = grid->xCoord[i];
            yCoord[i] = grid->yCoord[i];
            zCoord[i] = grid->zCoord[i];
            uData[i] = grid->uData[i];
        }
    }
    else
    {
        for (i = 0; i < grid->numPoints; i++)
        {
            xCoord[i] = grid->xCoord[i];
            yCoord[i] = grid->yCoord[i];
            zCoord[i] = grid->zCoord[i];
        }
    }

    // clean up
    for (i = 0; i < 6; i++)
        delete[] elementSide[i];
    delete[] elementProcessed;

    // done
    return (returnObject);
}

void Grid::regularTransformation(Grid *grid)
{
    (void)grid;
    fprintf(stderr, "\n\nNOT IMPLEMENTED !!!\a\a\a\n\n\n");
    return;
}

int Grid::areIdentical(int pl1[4], int pl2[4])
{
    int i, k;
    int r = 0;

    for (i = 0; i < 4; i++)
        for (k = 0; k < 4; k++)
            if (pl1[i] == pl2[k])
                r++;

    if (r != 4)
        r = 0;

    // done
    return (r);
}

int Grid::getStart(int el, int side)
{
    int r = -1;

    switch (typeList[el])
    {
    case TYPE_HEXAGON:
        switch (side)
        {
        case 0:
            r = connList[elemList[el]];
            break;
        case 1:
            r = connList[elemList[el] + 2];
            break;
        case 2:
            r = connList[elemList[el] + 4];
            break;
        case 3:
            r = connList[elemList[el]];
            break;
        case 4:
            r = connList[elemList[el]];
            break;
        case 5:
            r = connList[elemList[el] + 1];
            break;
        }
        break;
    case TYPE_PYRAMID:
        r = connList[elemList[el]];
        break;
    case TYPE_PRISM:
        switch (side)
        {
        case 0:
            r = connList[elemList[el]];
            break;
        case 1:
            r = connList[elemList[el] + 1];
            break;
        case 2:
            r = connList[elemList[el]];
            break;
        }
        break;
    }

    // done
    return (r);
}

int Grid::haveSameNeighbor(int el, int c0, int c1, int c2, int c3)
{
    int cCheck[4];
    int numCheck;

    int *neighbors[4];
    int numNeigh[4];

    int i, k, t = 0, j;
    int r = -1;
    int n;

    // initialize
    cCheck[0] = c0;
    cCheck[1] = c1;
    if (c2 == -1 || c3 == -1)
        numCheck = 2;
    else
    {
        cCheck[2] = c2;
        cCheck[3] = c3;
        numCheck = 4;
    }

    // get neighbors from neighborList
    for (i = 0; i < numCheck; i++)
    {
        neighbors[i] = (int *)(neighborList + neighborIndexList[cCheck[i]]);
        if (cCheck[i] == numPoints - 1)
            // we're at the end of the list
            n = numNeighbors;
        else
            n = neighborIndexList[cCheck[i] + 1];
        numNeigh[i] = n - neighborIndexList[cCheck[i]];
    }

    // now see, if they all share one neighbor (other then the given element el)
    for (k = 0; k < numNeigh[0] && r == -1; k++)
    {
        // the number of points that share the neighbor is given in t
        t = 1;

        // work through all points and theire neighbors
        for (i = 1; i < numCheck; i++)
            for (j = 0; j < numNeigh[i]; j++)
                if ((neighbors[i][j] == neighbors[0][k]) && neighbors[0][k] != el)
                    t++;

        // see if we're finished
        if (t == numCheck)
            r = neighbors[0][k];
    }

    // check on which side of the neighbor we are
    if (t == 4)
        onWhichSide = findSharedSide(r, cCheck[0], cCheck[1], cCheck[2], cCheck[3]);
    else
        onWhichSide = 0;

    // done
    return (r);
}

int Grid::findSharedSide(int el, int c0, int c1, int c2, int c3)
{
    int pl0[4], pl1[4];
    int r = -1;

    // init
    pl0[0] = c0;
    pl0[1] = c1;
    pl0[2] = c2;
    pl0[3] = c3;

    // work through given element
    switch (typeList[el])
    {
    case TYPE_HEXAGON:
        pl1[0] = connList[elemList[el] + 0];
        pl1[1] = connList[elemList[el] + 1];
        pl1[2] = connList[elemList[el] + 2];
        pl1[3] = connList[elemList[el] + 3];
        // check top
        if (areIdentical(pl0, pl1))
            r = 0;
        else
        {
            // check right
            pl1[0] = connList[elemList[el] + 2];
            pl1[1] = connList[elemList[el] + 6];
            pl1[2] = connList[elemList[el] + 7];
            pl1[3] = connList[elemList[el] + 3];
            if (areIdentical(pl0, pl1))
                r = 1;
            else
            {
                // check bottom
                pl1[0] = connList[elemList[el] + 4];
                pl1[1] = connList[elemList[el] + 7];
                pl1[2] = connList[elemList[el] + 6];
                pl1[3] = connList[elemList[el] + 5];
                if (areIdentical(pl0, pl1))
                    r = 2;
                else
                {
                    // check left
                    pl1[0] = connList[elemList[el] + 0];
                    pl1[1] = connList[elemList[el] + 4];
                    pl1[2] = connList[elemList[el] + 5];
                    pl1[3] = connList[elemList[el] + 1];
                    if (areIdentical(pl0, pl1))
                        r = 3;
                    else
                    {
                        // check front
                        pl1[0] = connList[elemList[el] + 0];
                        pl1[1] = connList[elemList[el] + 3];
                        pl1[2] = connList[elemList[el] + 4];
                        pl1[3] = connList[elemList[el] + 7];
                        if (areIdentical(pl0, pl1))
                            r = 4;
                        else
                        {
                            // check rear
                            pl1[0] = connList[elemList[el] + 1];
                            pl1[1] = connList[elemList[el] + 5];
                            pl1[2] = connList[elemList[el] + 6];
                            pl1[3] = connList[elemList[el] + 2];
                            if (areIdentical(pl0, pl1))
                                r = 5;
                        }
                    }
                }
            }
        }
        break;
    case TYPE_PYRAMID:
        // this is trivial
        r = 0;
        break;
    case TYPE_PRISM:
        pl1[0] = connList[elemList[el] + 0];
        pl1[1] = connList[elemList[el] + 2];
        pl1[2] = connList[elemList[el] + 5];
        pl1[3] = connList[elemList[el] + 3];
        // check front
        if (areIdentical(pl0, pl1))
            r = 0;
        else
        {
            // check right
            pl1[0] = connList[elemList[el] + 1];
            pl1[1] = connList[elemList[el] + 4];
            pl1[2] = connList[elemList[el] + 5];
            pl1[3] = connList[elemList[el] + 2];
            if (areIdentical(pl0, pl1))
                r = 1;
            else
            {
                // check left
                pl1[0] = connList[elemList[el] + 0];
                pl1[1] = connList[elemList[el] + 3];
                pl1[2] = connList[elemList[el] + 4];
                pl1[3] = connList[elemList[el] + 1];
                if (areIdentical(pl0, pl1))
                    r = 2;
            }
        }
        break;
    }

// debug
#ifdef DEBUG
    if (r == -1)
    {
        fprintf(stderr, "findSharedSide failed !!!\n");
        for (r = 0; r < 4; r++)
            fprintf(stderr, "g%d: %d\n", r, pl0[r]);
        for (r = 0; r < 8; r++)
            fprintf(stderr, "e%d: %d\n", r, connList[elemList[el] + r]);
        r = 0;
    }
#endif

    // done
    return (r);
}

int Grid::analyzeQuad(Grid *grid, int el, int v0, int v1, int v2, int v3)
{
    (void)grid;
    int r = -1;
    int n;
    //int start;

    n = haveSameNeighbor(el, v0, v1, v2, v3);
    if (n != -1)
    {
        // we are connected to another quad,
        // so we can do what we want (at least for now)
        r = -2;
    }
    else if (haveSameNeighbor(el, v0, v2))
        // the first diagonal is shared
        r = 1;
    else if (haveSameNeighbor(el, v1, v3))
        // the second diagonal is shared
        r = 2;

    // done
    return (r);
}

void Grid::analyzeSelf(Grid *grid, int &numSides, int &okSides, int &numHexa, int &numHexaOk)
{

    int i, j, n;
    int odc;

    numSides = okSides = 0;
    numHexa = numHexaOk = 0;

    for (i = 0; i < grid->numElem; i++)
    {
        n = 0;

        switch (grid->typeList[i])
        {
        case TYPE_HEXAGON:
            n = 6;
            numHexa++;

            odc = 0;

            // top/bottom
            odc += (elementSide[0][i] == elementSide[2][i]);

            // right/left
            if (elementSide[1][i] == 1 && elementSide[3][i] == 2)
                odc++;
            else if (elementSide[1][i] == 2 && elementSide[3][i] == 1)
                odc++;

            // front/rear
            odc += (elementSide[4][i] == elementSide[5][i]);

            if (odc)
                numHexaOk++;

            break;
        case TYPE_PYRAMID:
            n = 1;
            break;
        case TYPE_PRISM:
            n = 3;
            break;
        }
        numSides += n;
        for (j = 0; j < n; j++)
            if (elementSide[j][i] > 0)
                okSides++;
    }

    // done
    return;
}

int Grid::isInsideGetData(int c0, int c1, int c2, int c3, float xP, float yP, float zP, float &uP, float &vP, float &wP)
{
    float px[3], p0[3], p1[3], p2[3], p3[3];
    int r = 0;

    float w, w0, w1, w2, w3;

    px[0] = xP;
    px[1] = yP;
    px[2] = zP;

    p0[0] = xCoord[c0];
    p0[1] = yCoord[c0];
    p0[2] = zCoord[c0];

    p1[0] = xCoord[c1];
    p1[1] = yCoord[c1];
    p1[2] = zCoord[c1];

    p2[0] = xCoord[c2];
    p2[1] = yCoord[c2];
    p2[2] = zCoord[c2];

    p3[0] = xCoord[c3];
    p3[1] = yCoord[c3];
    p3[2] = zCoord[c3];

    w = fabsf(tetraVolume(p0, p1, p2, p3));

    w0 = fabsf(tetraVolume(px, p1, p2, p3)) / w;
    w1 = fabsf(tetraVolume(p0, px, p2, p3)) / w;
    w2 = fabsf(tetraVolume(p0, p1, px, p3)) / w;
    w3 = fabsf(tetraVolume(p0, p1, p2, px)) / w;

    if (w0 + w1 + w2 + w3 <= 1.00001)
    {
        // is inside !
        r = 1;

        uP = uData[c0] * w0 + uData[c1] * w1 + uData[c2] * w2 + uData[c3] * w3;
        if (vectorFlag)
        {
            vP = vData[c0] * w0 + vData[c1] * w1 + vData[c2] * w2 + vData[c3] * w3;
            wP = wData[c0] * w0 + wData[c1] * w1 + wData[c2] * w2 + wData[c3] * w3;
        }
    }

    // done
    return (r);
}

float Grid::tetraVolume(float p0[3], float p1[3], float p2[3], float p3[3])
{
    float v;

    v = (((p2[1] - p0[1]) * (p3[2] - p0[2]) - (p3[1] - p0[1]) * (p2[2] - p0[2])) * (p1[0] - p0[0]) + ((p2[2] - p0[2]) * (p3[0] - p0[0]) - (p3[2] - p0[2]) * (p2[0] - p0[0])) * (p1[1] - p0[1]) + ((p2[0] - p0[0]) * (p3[1] - p0[1]) - (p3[0] - p0[0]) * (p2[1] - p0[1])) * (p1[2] - p0[2])) / 6.0f;

    return (v);
}
