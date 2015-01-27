/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <appl/ApplInterface.h>
#include <do/coDoData.h>
#include <do/coDoUnstructuredGrid.h>
#include "TetraVolume.h"

#ifndef _WIN32
#include <unistd.h>
#endif
//////
////// we must provide main to init covise
//////

int main(int argc, char *argv[])
{
    // init
    new TetraVolume(argc, argv);

    // and back to covise
    Covise::main_loop();

    // done
    return 1;
}

coDistributedObject **TetraVolume::compute(const coDistributedObject **in, char **outNames)
{
    coDistributedObject **returnObject = NULL;
    coDoFloat *volOut = NULL;
    coDoUnstructuredGrid *gridIn = NULL;

    int numElem, numConn, numPoints;
    float *xCoord, *yCoord, *zCoord;
    int *elemList, *connList, *typeList;
    float *dataOut;

    int i;

    float p0[3], p1[3], p2[3], p3[3];

    // get input
    gridIn = (coDoUnstructuredGrid *)in[0];

    gridIn->getGridSize(&numElem, &numConn, &numPoints);
    gridIn->getAddresses(&elemList, &connList, &xCoord, &yCoord, &zCoord);
    gridIn->getTypeList(&typeList);

    // create output
    volOut = new coDoFloat(outNames[0], numElem);
    volOut->getAddress(&dataOut);

    // here we go
    for (i = 0; i < numElem; i++)
    {
        if (typeList[i] == TYPE_TETRAHEDER)
        {
            // do it
            p0[0] = xCoord[connList[elemList[i]]];
            p0[1] = yCoord[connList[elemList[i]]];
            p0[2] = zCoord[connList[elemList[i]]];
            p1[0] = xCoord[connList[elemList[i] + 1]];
            p1[1] = yCoord[connList[elemList[i] + 1]];
            p1[2] = zCoord[connList[elemList[i] + 1]];
            p2[0] = xCoord[connList[elemList[i] + 2]];
            p2[1] = yCoord[connList[elemList[i] + 2]];
            p2[2] = zCoord[connList[elemList[i] + 2]];
            p3[0] = xCoord[connList[elemList[i] + 3]];
            p3[1] = yCoord[connList[elemList[i] + 3]];
            p3[2] = zCoord[connList[elemList[i] + 3]];

            dataOut[i] = calculateVolume(p0, p1, p2, p3);

            /*
         if( dataOut[i] < 0.0 )
         {
            fprintf(stderr, "cell %d has volume %f\n", i, dataOut[i]);
            sleep( 2 );
         }
         */
        }
        else
            dataOut[i] = 1.0; // just so we get no div by zero
    }

    // yep
    returnObject = new coDistributedObject *[1];
    returnObject[0] = volOut;

    // done
    return (returnObject);
}

float TetraVolume::calculateVolume(float p0[3], float p1[3], float p2[3], float p3[3])
{
    float v;

    v = (((p2[1] - p0[1]) * (p3[2] - p0[2]) - (p3[1] - p0[1]) * (p2[2] - p0[2])) * (p1[0] - p0[0]) + ((p2[2] - p0[2]) * (p3[0] - p0[0]) - (p3[2] - p0[2]) * (p2[0] - p0[0])) * (p1[1] - p0[1]) + ((p2[0] - p0[0]) * (p3[1] - p0[1]) - (p3[0] - p0[0]) * (p2[1] - p0[1])) * (p1[2] - p0[2])) / 6.0f;

    return (v);
}
