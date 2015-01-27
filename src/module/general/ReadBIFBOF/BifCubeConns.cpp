/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// IMPLEMENTATION   BifCubeConns
//
// Description: Class for cubical connections
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

#include "BifCubeConns.h"

using namespace std;

//----------------------------------------------------------------------
//  KONSTUKTOR
//----------------------------------------------------------------------
BifCubeConns::BifCubeConns(int dsID, int pMinNodeID, int pMinID, int pMaxID, BifBof *pbifBof)
    : BifElement(dsID)
    , minID(pMinID)
    , maxID(pMaxID)
    , bifBof(pbifBof)
{
    numCubes = pMaxID - pMinID + 1;
    //    numConns  = numCubes*3;

    //   elem_list = new int[numCubes];
    //   conn_list = new int[numConns];
    minNodeID = pMinNodeID;
}

//----------------------------------------------------------------------
int BifCubeConns::readInConnections(BifNodalPoints *nodPoints, int &readingComplete)
{

    // get the connectivities of the record buffer
    for (int i = 0; i < numCubes; i++)
    {
        int ret = bifBof->readRegularRecord(elementBuffer, readingComplete);

        if (ret != 0)
            return ret;

        /* Transfer data from the entry buffer */
        /* to the local data structure */

        //       conn_list[8*i]   = nodPoints->getCoviceIdOfBifId(elementBuffer[5].i);
        //       conn_list[8*i+1] = nodPoints->getCoviceIdOfBifId(elementBuffer[8].i);
        //       conn_list[8*i+2] = nodPoints->getCoviceIdOfBifId(elementBuffer[7].i);
        //       conn_list[8*i+3] = nodPoints->getCoviceIdOfBifId(elementBuffer[6].i);
        //       conn_list[8*i+4] = nodPoints->getCoviceIdOfBifId(elementBuffer[1].i);
        //       conn_list[8*i+5] = nodPoints->getCoviceIdOfBifId(elementBuffer[4].i);
        //       conn_list[8*i+6] = nodPoints->getCoviceIdOfBifId(elementBuffer[3].i);
        //       conn_list[8*i+7] = nodPoints->getCoviceIdOfBifId(elementBuffer[2].i);
        conn_list.push_back(nodPoints->getCoviceIdOfBifId(elementBuffer[5].i));
        conn_list.push_back(nodPoints->getCoviceIdOfBifId(elementBuffer[8].i));
        conn_list.push_back(nodPoints->getCoviceIdOfBifId(elementBuffer[7].i));
        conn_list.push_back(nodPoints->getCoviceIdOfBifId(elementBuffer[6].i));
        conn_list.push_back(nodPoints->getCoviceIdOfBifId(elementBuffer[1].i));
        conn_list.push_back(nodPoints->getCoviceIdOfBifId(elementBuffer[4].i));
        conn_list.push_back(nodPoints->getCoviceIdOfBifId(elementBuffer[3].i));
        conn_list.push_back(nodPoints->getCoviceIdOfBifId(elementBuffer[2].i));

        if (readingComplete != 0)
            return 0;
    }

    return 0;
}

//----------------------------------------------------------------------
vector<int> *BifCubeConns::getCoviseConnections()
{
    return &conn_list;
}

//----------------------------------------------------------------------
vector<int> *BifCubeConns::getCoviseElementList()
{
    if (elem_list.size() == 0)
    {
        for (int i = 0; i < numCubes; i++)
        {
            //elem_list[i] = 8*i;
            elem_list.push_back(8 * i);
        }
    }
    return &elem_list;
}

//----------------------------------------------------------------------
int BifCubeConns::getNumConnections()
{
    return conn_list.size();
}

//----------------------------------------------------------------------
int BifCubeConns::getNumCubes()
{
    return numCubes;
}
