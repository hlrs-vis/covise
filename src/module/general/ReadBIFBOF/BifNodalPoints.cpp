/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// IMPLEMENTATION   BifNodalPoints
//
// Description: Class for the coordinates in nodal points
//              derived from BifElement
//
// Initial version: 06.2008
//
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2008 by Visenso
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
// Changes:
//

#include "BifNodalPoints.h"

using namespace std;

//----------------------------------------------------------------------
//  KONSTUKTOR
//----------------------------------------------------------------------
BifNodalPoints::BifNodalPoints(int dsID, int _minID, int _maxID, int numCoords, BifBof *pbifBof)
    : BifElement(dsID)
    , xcoords(NULL)
    , ycoords(NULL)
    , zcoords(NULL)
    , bifBof(pbifBof)
{
    //numCoord = pMaxID-pMinID+1;//andi:das ist falsch!
    numCoord = numCoords;
    minID = _minID;
    maxID = _maxID;
    xcoords = new float[numCoord];
    ycoords = new float[numCoord];
    zcoords = new float[numCoord];
}

//----------------------------------------------------------------------
// Transfer data from the entry buffer to the local data structure
int BifNodalPoints::readInCoordinates(int &readingComplete)
{

    // get the coordinates of the record buffer
    for (int i = 0; i < numCoord; i++)
    {
        int ret = bifBof->readRegularRecord(elementBuffer, readingComplete);

        if (ret != 0)
            return ret;

        //the node ID doesnt have to start at 0
        bifToCovID[elementBuffer[0].i] = i;

        xcoords[i] = elementBuffer[1].f;
        ycoords[i] = elementBuffer[2].f;
        zcoords[i] = elementBuffer[3].f;
        //cout << "biffID: " << elementBuffer[0].i << "  coords[" << i << "]" << xcoords[i] <<"|"<< ycoords[i]<<"|"<< zcoords[i]<< endl;
        //TODO: COORDINATE SYSTEM!?!?
        //nodalPoints[n].coordSysID[0]  = buffer[4].f;
        //nodalPoints[n].coordSysID[1]  = buffer[5].f;
        //nodalPoints[n].coordSysID[2]  = buffer[6].f;

        // done
        if (readingComplete != 0)
        {
            return 0;
        }
    }
    return 0;
}

//----------------------------------------------------------------------
int BifNodalPoints::getCoviseCoordinates(float **x_covise_coords, float **y_covise_coords, float **z_covise_coords)
{
    if (xcoords && ycoords && zcoords)
    {
        *x_covise_coords = xcoords;
        *y_covise_coords = ycoords;
        *z_covise_coords = zcoords;
    }
    else
    {
        return 102;
    }
    return 0;
}

//----------------------------------------------------------------------
int BifNodalPoints::getCoviceIdOfBifId(int BifID)
{
    map<int, int>::iterator it = bifToCovID.find(BifID);
    if (it != bifToCovID.end()) // the nodeID might not exist int the bif file return -1 if so
        return it->second;
    return -BifID;
}

//----------------------------------------------------------------------
int BifNodalPoints::getNumCoordinates()
{
    return numCoord;
}
