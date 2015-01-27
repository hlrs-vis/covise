/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// IMPLEMENTATION   BifPyramidConns
//
// Description: Class for pyramid connections
//              derived from BifElement
//
// Initial version: 11.2008
//
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2008 by Visenso
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
// Changes:
//

#include "BifPyramidConns.h"

using namespace std;

//----------------------------------------------------------------------
//  KONSTUKTOR
//----------------------------------------------------------------------
BifPyramidConns::BifPyramidConns(int dsID, int pMinNodeID, int pMinID, int pMaxID, BifBof *pbifBof)
    : BifElement(dsID)
    , minID(pMinID)
    , maxID(pMaxID)
    , bifBof(pbifBof)
{
    numPyramids = pMaxID - pMinID + 1;
    //    numConns  = numPyramids*5;

    //   elem_list = new int[numPyramids];
    //   conn_list = new int[numConns];
    minNodeID = pMinNodeID;
}

//----------------------------------------------------------------------
int BifPyramidConns::readInConnections(BifNodalPoints *nodPoints, int &readingComplete)
{

    // get the connectivities of the record buffer
    for (int i = 0; i < numPyramids; i++)
    {
        int ret = bifBof->readRegularRecord(elementBuffer, readingComplete);

        if (ret != 0)
            return ret;

        // Transfer data from the entry buffer to the local data structure

        //       conn_list[5*i]   = nodPoints->getCoviceIdOfBifId(elementBuffer[1].i);
        //       conn_list[5*i+1] = nodPoints->getCoviceIdOfBifId(elementBuffer[2].i);
        //       conn_list[5*i+2] = nodPoints->getCoviceIdOfBifId(elementBuffer[3].i);
        //       conn_list[5*i+3] = nodPoints->getCoviceIdOfBifId(elementBuffer[4].i);
        //       conn_list[5*i+4] = nodPoints->getCoviceIdOfBifId(elementBuffer[5].i);
        conn_list.push_back(nodPoints->getCoviceIdOfBifId(elementBuffer[1].i));
        conn_list.push_back(nodPoints->getCoviceIdOfBifId(elementBuffer[2].i));
        conn_list.push_back(nodPoints->getCoviceIdOfBifId(elementBuffer[3].i));
        conn_list.push_back(nodPoints->getCoviceIdOfBifId(elementBuffer[4].i));
        conn_list.push_back(nodPoints->getCoviceIdOfBifId(elementBuffer[5].i));

        if (readingComplete != 0)
            return 0;
    }

    return 0;
}

//----------------------------------------------------------------------
vector<int> *BifPyramidConns::getCoviseConnections()
{
    return &conn_list;
}

//----------------------------------------------------------------------
vector<int> *BifPyramidConns::getCoviseElementList()
{
    if (elem_list.size() == 0)
    {
        for (int i = 0; i < numPyramids; i++)
        {
            //elem_list[i] = 5*i;
            elem_list.push_back(5 * i);
        }
    }
    return &elem_list;
}

//----------------------------------------------------------------------
int BifPyramidConns::getNumConnections()
{
    return conn_list.size();
}

//----------------------------------------------------------------------
int BifPyramidConns::getNumPyramids()
{
    return numPyramids;
}
