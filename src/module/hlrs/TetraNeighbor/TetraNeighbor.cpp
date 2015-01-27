/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <appl/ApplInterface.h>
#include <do/coDoSet.h>
#include <do/coDoData.h>
#include <do/coDoUnstructuredGrid.h>
#include "TetraNeighbor.h"

//////
////// we must provide main to init covise
//////

int main(int argc, char *argv[])
{
    // init
    new TetraNeighbor(argc, argv);

    // and back to covise
    Covise::main_loop();

    // done
    return 0;
}

coDistributedObject **TetraNeighbor::compute(const coDistributedObject **in, char **outNames)
{
    coDistributedObject **returnObject = NULL;
    const coDoUnstructuredGrid *gridIn;
    coDoFloat *data;
    const coDistributedObject *const *setIn = NULL;
    coDoFloat **setOut;
    coDoSet *outputSet;

    int numElem, numConn, numPoints;

    float *newNeighborList;

    int i, n;

    char bfr[1024];

    // prepare output
    returnObject = new coDistributedObject *[1];
    returnObject[0] = NULL;

    // the neighborlist contains 5 entries for each element:
    //   0:  side 123
    //   1:  side 142
    //   2:  side 134
    //   3:  side 243
    //   4:  cell in next timestep, which contains this cells centroid
    // any of the values may be set to -1 if no neighbor/cell exists

    const char *dataType = (in[0])->getType();

    if (!strcmp(dataType, "SETELE"))
    {
        // SET !!!
        setIn = ((const coDoSet *)in[0])->getAllElements(&n);

        // init output
        setOut = new coDoFloat *[n + 1];
        setOut[n] = NULL;
        for (i = 0; i < n; i++)
        {
            gridIn = (const coDoUnstructuredGrid *)setIn[i];
            gridIn->getGridSize(&numElem, &numConn, &numPoints);
            sprintf(bfr, "%s_%d", outNames[0], i);
            setOut[i] = new coDoFloat(bfr, numElem * 5);
        }

// first create the neighbor-list for each timestep
#if !defined(__sgi)
        int j;
        for (j = 0; j < n; j++)
        {
            gridIn = (const coDoUnstructuredGrid *)setIn[j];
            setOut[j]->getAddress(&newNeighborList);
            if (!localNeighbors(gridIn, newNeighborList, -4.0))
                return NULL;

            fprintf(stderr, "TetraNeighbor::compute: localNeighbors(%d) done\n", j);
        }
#else
        goSMP(setIn, setOut, n, m_numNodes);
#endif
        /*
      // now compute cross-timestep-neighbor-information
      for( i=0; i<n; i++ )
      {
         if( i==n-1 )
       j = 0;
      else
       j = i+1;

      setOut[i]->getAddress( &newNeighborList );
      setOut[j]->getAddress( &nextNeighborList );

      timeNeighbors( (coDoUnstructuredGrid *)setIn[i], newNeighborList, \ 
      (coDoUnstructuredGrid *)setIn[j], nextNeighborList );

      fprintf(stderr, "TetraNeighbor::compute: timeNeighbors(%d, %d) done\n", i, j);
      }
      */

        // build output
        outputSet = new coDoSet(outNames[0], (coDistributedObject **)setOut);
        copyAttributes(in[0], outputSet);

        // clean up
        for (i = 0; setOut[i]; i++)
            delete setOut[i];
        delete[] setOut;

        returnObject[0] = outputSet;
    }
    else
    {
        // get input
        gridIn = (coDoUnstructuredGrid *)in[0];
        gridIn->getGridSize(&numElem, &numConn, &numPoints);

        // create output
        data = new coDoFloat(outNames[0], numElem * 5);
        data->getAddress(&newNeighborList);

        if (!localNeighbors(gridIn, newNeighborList))
            return NULL;

        for (i = 0; i < numElem; i++)
            newNeighborList[(i * 5) + 4] = (float)i;

        // done
        returnObject[0] = data;
    }

    // back
    return (returnObject);
}

int TetraNeighbor::localNeighbors(const coDoUnstructuredGrid *grid, float *newNeighborList, float tval)
{
    int i;
    int numElem, numConn, numPoints;
    int *typeList, *elemList, *connList;
    int *neighborList, *neighborIndexList;
    float *xCoord, *yCoord, *zCoord;
    int numNeighbors;

    int o;
    int c0, c1, c2, c3;

    grid->getGridSize(&numElem, &numConn, &numPoints);
    grid->getAddresses(&elemList, &connList, &xCoord, &yCoord, &zCoord);
    grid->getTypeList(&typeList);

    grid->getNeighborList(&numNeighbors, &neighborList, &neighborIndexList);

    // compute output
    for (i = 0; i < numElem; i++)
    {
        if (typeList[i] != TYPE_TETRAHEDER)
        {
            Covise::sendError("gridIn contains other elements then tetrahedra");
            return 0;
        }
        o = elemList[i];
        c0 = connList[o];
        c1 = connList[o + 1];
        c2 = connList[o + 2];
        c3 = connList[o + 3];

        o = i * 5;

        newNeighborList[o] = getNeighbor(i, c0, c1, c2, numNeighbors, neighborList, neighborIndexList, numPoints);
        newNeighborList[o + 1] = getNeighbor(i, c0, c3, c1, numNeighbors, neighborList, neighborIndexList, numPoints);
        newNeighborList[o + 2] = getNeighbor(i, c0, c2, c3, numNeighbors, neighborList, neighborIndexList, numPoints);
        newNeighborList[o + 3] = getNeighbor(i, c1, c3, c2, numNeighbors, neighborList, neighborIndexList, numPoints);
        //newNeighborList[o+4] = -1.0;
        newNeighborList[o + 4] = tval;
    }

    // done
    return 1;
}

float TetraNeighbor::getNeighbor(int el, int c0, int c1, int c2, int numNeighbors, int *neighborList, int *neighborIndexList, int numPoints)
{
    int *neighbors[3];
    int numNeigh[3];

    int i, j, k;
    int n;
    int t;

    float r;

    neighbors[0] = (int *)(neighborList + neighborIndexList[c0]);
    neighbors[1] = (int *)(neighborList + neighborIndexList[c1]);
    neighbors[2] = (int *)(neighborList + neighborIndexList[c2]);

    if (c0 == numPoints - 1)
        n = numNeighbors;
    else
        n = neighborIndexList[c0 + 1];
    numNeigh[0] = n - neighborIndexList[c0];

    if (c1 == numPoints - 1)
        n = numNeighbors;
    else
        n = neighborIndexList[c1 + 1];
    numNeigh[1] = n - neighborIndexList[c1];

    if (c2 == numPoints - 1)
        n = numNeighbors;
    else
        n = neighborIndexList[c2 + 1];
    numNeigh[2] = n - neighborIndexList[c2];

    // here we go
    n = -1;
    for (i = 0; i < numNeigh[0] && n == -1; i++)
    {
        t = 1;
        for (k = 1; k < 3; k++)
            for (j = 0; j < numNeigh[k]; j++)
                if ((neighbors[k][j] == neighbors[0][i]) && neighbors[0][i] != el)
                {
                    t++;
                    if (t == 3)
                    {
                        n = neighbors[0][i];
                        k = 3;
                        //break;
                    }
                    break;
                }
    }

    r = (float)n;

    // done
    return (r);
}

void TetraNeighbor::timeNeighbors(const coDoUnstructuredGrid *grid, float *neighborList,
                                  coDoUnstructuredGrid *nextGrid, float *nextNeighborList)
{
    (void)nextNeighborList;
    //fprintf(stderr, "timeNeighbors: poorly implemented !\n");

    //float cs[3]; // centroid of current cell
    int i, j;
    int o;
    int c0, c1, c2, c3;
    int found;

    int numElem[2], numConn[2], numPoints[2];
    int *elemList[2], *connList[2];
    float *xCoord[2], *yCoord[2], *zCoord[2];
    float p0[3], p1[3], p2[3], p3[3], px[3];

    float range, d;

    // get data
    grid->getGridSize(&numElem[0], &numConn[0], &numPoints[0]);
    grid->getAddresses(&elemList[0], &connList[0], &xCoord[0], &yCoord[0], &zCoord[0]);

    nextGrid->getGridSize(&numElem[1], &numConn[1], &numPoints[1]);
    nextGrid->getAddresses(&elemList[1], &connList[1], &xCoord[1], &yCoord[1], &zCoord[1]);

    // we have to find a start
    found = 0;
    for (i = 0; i < numElem[0] && !found; i++)
    {
        o = elemList[0][i];
        c0 = connList[0][o];
        c1 = connList[0][o + 1];
        c2 = connList[0][o + 2];
        c3 = connList[0][o + 3];

        p0[0] = xCoord[0][c0];
        p0[1] = yCoord[0][c0];
        p0[2] = zCoord[0][c0];

        p1[0] = xCoord[0][c1];
        p1[1] = yCoord[0][c1];
        p1[2] = zCoord[0][c1];

        p2[0] = xCoord[0][c2];
        p2[1] = yCoord[0][c2];
        p2[2] = zCoord[0][c2];

        p3[0] = xCoord[0][c3];
        p3[1] = yCoord[0][c3];
        p3[2] = zCoord[0][c3];
        range = powf(tetraVolume(p0, p1, p2, p3), 2.0f / 3.0f);
        range *= 36.0f;
        //range *= range;

        px[0] = (xCoord[0][c0] + xCoord[0][c1] + xCoord[0][c2] + xCoord[0][c3]) / 4.0f;
        px[1] = (yCoord[0][c0] + yCoord[0][c1] + yCoord[0][c2] + yCoord[0][c3]) / 4.0f;
        px[2] = (zCoord[0][c0] + zCoord[0][c1] + zCoord[0][c2] + zCoord[0][c3]) / 4.0f;

        for (j = 0; j < numElem[1] && !found; j++)
        {
            o = elemList[1][j];
            c0 = connList[1][o];

            p0[0] = xCoord[1][c0];
            p0[1] = yCoord[1][c0];
            p0[2] = zCoord[1][c0];

            p1[0] = px[0] - p0[0];
            p1[1] = px[1] - p0[1];
            p1[2] = px[2] - p0[2];

            d = p1[0] * p1[0] + p1[1] * p1[1] + p1[2] * p1[2];

            if (d < range)
            {
                c1 = connList[1][o + 1];
                c2 = connList[1][o + 2];
                c3 = connList[1][o + 3];

                p1[0] = xCoord[1][c1];
                p1[1] = yCoord[1][c1];
                p1[2] = zCoord[1][c1];

                p2[0] = xCoord[1][c2];
                p2[1] = yCoord[1][c2];
                p2[2] = zCoord[1][c2];

                p3[0] = xCoord[1][c3];
                p3[1] = yCoord[1][c3];
                p3[2] = zCoord[1][c3];

                if (isInCell(p0, p1, p2, p3, px))
                {
                    //found = 1;
                    neighborList[(i * 5) + 4] = (float)j;
                }
            }
        }
    }
    //fprintf(stderr, "TetraNeighbor::timeNeighbors: %d\n", i);
    return;

    /* 
      // ok, we have a start
      found = 0;
      while( !found )
      {
         found = 1;
         for( i=0; i<numElem[0]; i++ )
         {
            // has this cell been processed
            n = (int)neighborList[(i*5)+4];
            if( n >= 0 )
   {
   // yes, so work through neighbors
   n3 = 0;
   for( n2=0; n2<4 && !n3; n2++ )
   {

   }

   }
   else
   if( n==-4 )
   found=0;
   }
   }
   */

    /*

      // we have to make a start somewhere, so we do it with the first cell
      found = 0;
      for( i=0; i<numElem[0] && !found; i++ )
      {
       o = elemList[0][i];
       c0 = connList[0][o];
       c1 = connList[0][o+1];
       c2 = connList[0][o+2];
       c3 = connList[0][o+3];

   px[0] = (xCoord[0][c0]+xCoord[0][c1]+xCoord[0][c2]+xCoord[0][c3])/4.0;
   px[1] = (yCoord[0][c0]+yCoord[0][c1]+yCoord[0][c2]+yCoord[0][c3])/4.0;
   px[2] = (zCoord[0][c0]+zCoord[0][c1]+zCoord[0][c2]+zCoord[0][c3])/4.0;

   // maybe the cell doesnt move, so check for it
   o = elemList[1][0];
   c0 = connList[1][o];
   c1 = connList[1][o+1];
   c2 = connList[1][o+2];
   c3 = connList[1][o+3];

   p0[0] = xCoord[1][c0];
   p0[1] = yCoord[1][c0];
   p0[2] = zCoord[1][c0];

   p1[0] = xCoord[1][c1];
   p1[1] = yCoord[1][c1];
   p1[2] = zCoord[1][c1];

   p2[0] = xCoord[1][c2];
   p2[1] = yCoord[1][c2];
   p2[2] = zCoord[1][c2];

   p3[0] = xCoord[1][c3];
   p3[1] = yCoord[1][c3];
   p3[2] = zCoord[1][c3];

   if( isInCell( p0, p1, p2, p3, px ) )
   neighborList[4] = (float)0;
   else
   {
   // damn

   o = elemList[0][0];
   c0 = connList[0][o];
   c1 = connList[0][o+1];
   c2 = connList[0][o+2];
   c3 = connList[0][o+3];

   p0[0] = xCoord[0][c0];
   p0[1] = yCoord[0][c0];
   p0[2] = zCoord[0][c0];

   p1[0] = xCoord[0][c1];
   p1[1] = yCoord[0][c1];
   p1[2] = zCoord[0][c1];

   p2[0] = xCoord[0][c2];
   p2[1] = yCoord[0][c2];
   p2[2] = zCoord[0][c2];

   p3[0] = xCoord[0][c3];
   p3[1] = yCoord[0][c3];
   p3[2] = zCoord[0][c3];
   range = powf( tetraVolume(p0, p1, p2, p3), 2.0/3.0 );
   range *= 36.0;
   found = 0;
   //range *= range;

   for( j=0; j<numElem[1] && !found; j++ )
   {
   o = elemList[1][j];
   c0 = connList[1][o];

   p0[0] = xCoord[1][c0];
   p0[1] = yCoord[1][c0];
   p0[2] = zCoord[1][c0];

   p1[0] = px[0]-p0[0];
   p1[1] = px[1]-p0[1];
   p1[2] = px[2]-p0[2];

   d = p1[0]*p1[0] + p1[1]*p1[1] + p1[2]*p1[2];

   if( d < range )
   {

   c1 = connList[1][o+1];
   c2 = connList[1][o+2];
   c3 = connList[1][o+3];

   p1[0] = xCoord[1][c1];
   p1[1] = yCoord[1][c1];
   p1[2] = zCoord[1][c1];

   p2[0] = xCoord[1][c2];
   p2[1] = yCoord[1][c2];
   p2[2] = zCoord[1][c2];

   p3[0] = xCoord[1][c3];
   p3[1] = yCoord[1][c3];
   p3[2] = zCoord[1][c3];

   if( isInCell( p0, p1, p2, p3, px ) )
   {
   found = 1;
   neighborList[4] = (float)j;
   }

   }

   }

   if( found )
   {
   // fine

   fprintf(stderr, "TetraNeighbot::timeNeighbors: %d\n", j-1);

   }
   else
   {
   // no time-neighbors

   fprintf(stderr, "TetraNeighbor::timeNeighbors: no neighbors found\n");

   for( i=0; i<numElem[0]; i++ )
   neighborList[(i*5)+4]=-1.0;
   }

   }
   */
    /*

   // work through each cell
   for( i=0; i<numElem[0]; i++ )
   {
      // first compute centroid of current cell
      o = elemList[0][i];
      c0 = connList[0][o];
      c1 = connList[0][o+1];
      c2 = connList[0][o+2];
   c3 = connList[0][o+3];

   px[0] = (xCoord[0][c0]+xCoord[0][c1]+xCoord[0][c2]+xCoord[0][c3])/4.0;
   px[1] = (yCoord[0][c0]+yCoord[0][c1]+yCoord[0][c2]+yCoord[0][c3])/4.0;
   px[2] = (zCoord[0][c0]+zCoord[0][c1]+zCoord[0][c2]+zCoord[0][c3])/4.0;

   // now we have to find the cell containing the given point px
   found = 0;

   // does the cell exist in the "next" grid
   if( i < numElem[1] )
   {
   // check if we are in this cell
   o = elemList[1][i];
   c0 = connList[1][o];
   c1 = connList[1][o+1];
   c2 = connList[1][o+2];
   c3 = connList[1][o+3];

   p0[0] = xCoord[1][c0];
   p0[1] = yCoord[1][c0];
   p0[2] = zCoord[1][c0];

   p1[0] = xCoord[1][c1];
   p1[1] = yCoord[1][c1];
   p1[2] = zCoord[1][c1];

   p2[0] = xCoord[1][c2];
   p2[1] = yCoord[1][c2];
   p2[2] = zCoord[1][c2];

   p3[0] = xCoord[1][c3];
   p3[1] = yCoord[1][c3];
   p3[2] = zCoord[1][c3];

   if( isInCell( p0, p1, p2, p3, px ) )
   {
   found = 1;
   neighborList[(i*5)+4] = (float)i;
   }
   }
   if( !found )
   {
   // not yet found, so see if any of our neighbors
   // has allready found its neighbor in the next timestep
   n = -1;
   for( j=0; j<4 && !found; j++ )
   {
   n = (int)neighborList[(i*5)+j];

   if( neighborList[(n*5)+4]>=0.0 )
   {
   // this neighbor has a "time"-neighbor
   n = (int)neighborList[(n*5)+4];

   // work through neighbors of that cell after checking
   // the cell itself
   o = elemList[1][n];
   c0 = connList[1][o];
   c1 = connList[1][o+1];
   c2 = connList[1][o+2];
   c3 = connList[1][o+3];

   p0[0] = xCoord[1][c0];
   p0[1] = yCoord[1][c0];
   p0[2] = zCoord[1][c0];

   p1[0] = xCoord[1][c1];
   p1[1] = yCoord[1][c1];
   p1[2] = zCoord[1][c1];

   p2[0] = xCoord[1][c2];
   p2[1] = yCoord[1][c2];
   p2[2] = zCoord[1][c2];

   p3[0] = xCoord[1][c3];
   p3[1] = yCoord[1][c3];
   p3[2] = zCoord[1][c3];

   if( isInCell( p0, p1, p2, p3, px ) )
   {
   found = 1;
   neighborList[(i*5)+4] = (float)n;
   }
   else
   {
   // now work through the neighbors
   for( k=0; k<4 && !found; k++ )
   {
   n2 = (int)nextNeighborList[(n*5)+k];
   if( n2 != -1 )
   {
   // neighbor exists, so check it
   o = elemList[1][n2];
   c0 = connList[1][o];
   c1 = connList[1][o+1];
   c2 = connList[1][o+2];
   c3 = connList[1][o+3];

   p0[0] = xCoord[1][c0];
   p0[1] = yCoord[1][c0];
   p0[2] = zCoord[1][c0];

   p1[0] = xCoord[1][c1];
   p1[1] = yCoord[1][c1];
   p1[2] = zCoord[1][c1];

   p2[0] = xCoord[1][c2];
   p2[1] = yCoord[1][c2];
   p2[2] = zCoord[1][c2];

   p3[0] = xCoord[1][c3];
   p3[1] = yCoord[1][c3];
   p3[2] = zCoord[1][c3];

   if( isInCell( p0, p1, p2, p3, px ) )
   {
   found = 1;
   neighborList[(i*5)+4] = (float)n2;
   }
   }
   }
   }
   }
   }
   }
   if( !found )
   {
   // ok, we have to do an overall search
   //fprintf(stderr, "o");

   o = elemList[0][i];
   c0 = connList[0][o];
   c1 = connList[0][o+1];
   c2 = connList[0][o+2];
   c3 = connList[0][o+3];

   p0[0] = xCoord[0][c0];
   p0[1] = yCoord[0][c0];
   p0[2] = zCoord[0][c0];

   p1[0] = xCoord[0][c1];
   p1[1] = yCoord[0][c1];
   p1[2] = zCoord[0][c1];

   p2[0] = xCoord[0][c2];
   p2[1] = yCoord[0][c2];
   p2[2] = zCoord[0][c2];

   p3[0] = xCoord[0][c3];
   p3[1] = yCoord[0][c3];
   p3[2] = zCoord[0][c3];
   range = powf( tetraVolume(p0, p1, p2, p3), 2.0/3.0 );
   range *= 36.0;
   //range *= range;

   for( j=0; j<numElem[1] && !found; j++ )
   {
   o = elemList[1][j];
   c0 = connList[1][o];

   p0[0] = xCoord[1][c0];
   p0[1] = yCoord[1][c0];
   p0[2] = zCoord[1][c0];

   p1[0] = px[0]-p0[0];
   p1[1] = px[1]-p0[1];
   p1[2] = px[2]-p0[2];

   d = p1[0]*p1[0] + p1[1]*p1[1] + p1[2]*p1[2];

   if( d < range )
   {

   c1 = connList[1][o+1];
   c2 = connList[1][o+2];
   c3 = connList[1][o+3];

   p1[0] = xCoord[1][c1];
   p1[1] = yCoord[1][c1];
   p1[2] = zCoord[1][c1];

   p2[0] = xCoord[1][c2];
   p2[1] = yCoord[1][c2];
   p2[2] = zCoord[1][c2];

   p3[0] = xCoord[1][c3];
   p3[1] = yCoord[1][c3];
   p3[2] = zCoord[1][c3];

   if( isInCell( p0, p1, p2, p3, px ) )
   {
   found = 1;
   neighborList[(i*5)+4] = (float)j;
   }

   }

   }
   }

   // on to the next cell
   }
   */

    // done
    //return;
}

int TetraNeighbor::isInCell(float p0[3], float p1[3], float p2[3], float p3[3], float px[3])
{
    float w, w0, w1, w2, w3;
    int r = 0;

    w = fabsf(tetraVolume(p0, p1, p2, p3));

    w0 = fabsf(tetraVolume(px, p1, p2, p3)) / w;
    w1 = fabsf(tetraVolume(p0, px, p2, p3)) / w;
    w2 = fabsf(tetraVolume(p0, p1, px, p3)) / w;
    w3 = fabsf(tetraVolume(p0, p1, p2, px)) / w;

    if (w0 + w1 + w2 + w3 <= 1.0001)
        r = 1;

    // done
    return (r);
}

float TetraNeighbor::tetraVolume(float p0[3], float p1[3], float p2[3], float p3[3])
{
    float v;

    v = (((p2[1] - p0[1]) * (p3[2] - p0[2]) - (p3[1] - p0[1]) * (p2[2] - p0[2])) * (p1[0] - p0[0]) + ((p2[2] - p0[2]) * (p3[0] - p0[0]) - (p3[2] - p0[2]) * (p2[0] - p0[0])) * (p1[1] - p0[1]) + ((p2[0] - p0[0]) * (p3[1] - p0[1]) - (p3[0] - p0[0]) * (p2[1] - p0[1])) * (p1[2] - p0[2])) / 6.0f;

    return (v);
}
