/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// IMPLEMENTATION   BofScalarData
//
// Description: Class for the bof scalar temperatur
//              derived from BifElement
//
// Initial version: 10.2008
//
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2008 by HLRS
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
// Changes:
//

#include "BofScalarData.h"

using namespace std;

//----------------------------------------------------------------------
//  KONSTUKTOR
//----------------------------------------------------------------------
BofScalarData::BofScalarData(int dsID, int pCoordMinID, int pCoordMaxID, int pNumCoords, int pMinID, int pMaxID, BifBof *bb)
    : BifElement(dsID)
    , minID(pMinID)
    , maxID(pMaxID)
    , bifBof(bb)
{
    coordMinID = pCoordMinID;
    coordMaxID = pCoordMaxID;
    numCoord = pNumCoords;
    scalarvalue = new float[numCoord];
}

//----------------------------------------------------------------------
// Transfer data from the entry buffer to the local data structure
int BofScalarData::readScalarData(BifNodalPoints *nodPoints, int &readingComplete)
{
    // get the coordinates of the record buffer
    //numCoord=0;
    for (int i = minID; i <= maxID; i++)
    {
        int ret = bifBof->readRegularRecord(elementBuffer, readingComplete);

        if (ret != 0)
            return ret;

        if (elementBuffer[0].i >= coordMinID && elementBuffer[0].i <= coordMaxID)
        {

            //the node ID doesnt have to start at 0
            int nodeNum = nodPoints->getCoviceIdOfBifId(elementBuffer[0].i);
            if (nodeNum >= 0)
            {
                scalarvalue[nodeNum] = elementBuffer[1].f;
                //numCoord++;
            }
        }
        if (elementBuffer[0].i > coordMaxID)
            break;
    }
    return 0;
}
//----------------------------------------------------------------------
float *BofScalarData::getScalarArray()
{
    return scalarvalue;
}
//----------------------------------------------------------------------
int BofScalarData::getNumCoordinates()
{
    return numCoord;
}
