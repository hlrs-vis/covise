/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// IMPLEMENTATION   BifQuadConns
//
// Description: Class for quadrilateral connections
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

#include "BifQuadConns.h"

using namespace std;

//----------------------------------------------------------------------
//  KONSTUKTOR
//----------------------------------------------------------------------
BifQuadConns::BifQuadConns(int dsID, int pMinNodeID, int pMinID, int pMaxID, BifBof *pbifBof)
    : BifElement(dsID)
    , minID(pMinID)
    , maxID(pMaxID)
    , bifBof(pbifBof)
{
    numQuads = pMaxID - pMinID + 1;
    //    numConns  = numQuad*4;

    //elem_list = new int[numQuads];
    //conn_list = new int[numConns];
    minNodeID = pMinNodeID;
}

//----------------------------------------------------------------------
int BifQuadConns::readInConnections(BifNodalPoints *nodPoints, int &readingComplete)
{

    // get the connectivities of the record buffer
    for (int i = 0; i < numQuads; i++)
    {
        int ret = bifBof->readRegularRecord(elementBuffer, readingComplete);

        if (ret != 0)
            return ret;

        // Transfer data from the entry buffer to the local data structure

        /* Be aware that the structure of covise (Reference: Programming Guide)
      *  and Bif is different
      */

        //       conn_list[4*i]   = nodPoints->getCoviceIdOfBifId(elementBuffer[1].i);
        //       conn_list[4*i+1] = nodPoints->getCoviceIdOfBifId(elementBuffer[4].i);
        //       conn_list[4*i+2] = nodPoints->getCoviceIdOfBifId(elementBuffer[3].i);
        //       conn_list[4*i+3] = nodPoints->getCoviceIdOfBifId(elementBuffer[2].i);
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
vector<int> *BifQuadConns::getCoviseConnections()
{
    return &conn_list;
}

//----------------------------------------------------------------------
vector<int> *BifQuadConns::getCoviseElementList()
{
    if (elem_list.size() == 0)
    {
        for (int i = 0; i < numQuads; i++)
        {
            //elem_list[i] = 4*i;
            elem_list.push_back(4 * i);
        }
    }
    return &elem_list;
}

//----------------------------------------------------------------------
int BifQuadConns::getNumConnections()
{
    return conn_list.size();
}

//----------------------------------------------------------------------
int BifQuadConns::getNumQuads()
{
    return numQuads;
}
