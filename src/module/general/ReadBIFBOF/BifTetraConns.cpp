/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// IMPLEMENTATION   BifTetraConns
//
// Description: Class for tetrahedron connections
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

#include "BifTetraConns.h"

using namespace std;

//----------------------------------------------------------------------
//  KONSTUKTOR
//----------------------------------------------------------------------
BifTetraConns::BifTetraConns(int dsID, int pMinNodeID, int pMinID, int pMaxID, BifBof *pbifBof)
    : BifElement(dsID)
    , minID(pMinID)
    , maxID(pMaxID)
    , bifBof(pbifBof)
{
    numTetras = pMaxID - pMinID + 1;
    //numConns  = numTetras*4;

    //   elem_list = new int[numTetras];
    //   conn_list = new int[numConns];
    minNodeID = pMinNodeID;
}

//----------------------------------------------------------------------
int BifTetraConns::readInConnections(BifNodalPoints *nodPoints, int &readingComplete)
{

    // get the connectivities of the record buffer
    for (int i = 0; i < numTetras; i++)
    {
        int ret = bifBof->readRegularRecord(elementBuffer, readingComplete);

        if (ret != 0)
            return ret;

        // Transfer data from the entry buffer to the local data structure

        /* Be aware that the structure of covise (Reference: Programming Guide)
       * and Bif is different
       */
        //       conn_list[4*i]   = nodPoints->getCoviceIdOfBifId(elementBuffer[1].i);
        //       conn_list[4*i+1] = nodPoints->getCoviceIdOfBifId(elementBuffer[2].i);
        //       conn_list[4*i+2] = nodPoints->getCoviceIdOfBifId(elementBuffer[3].i);
        //       conn_list[4*i+3] = nodPoints->getCoviceIdOfBifId(elementBuffer[4].i);
        conn_list.push_back(nodPoints->getCoviceIdOfBifId(elementBuffer[1].i));
        conn_list.push_back(nodPoints->getCoviceIdOfBifId(elementBuffer[2].i));
        conn_list.push_back(nodPoints->getCoviceIdOfBifId(elementBuffer[3].i));
        conn_list.push_back(nodPoints->getCoviceIdOfBifId(elementBuffer[4].i));

        if (readingComplete != 0)
            return 0;
    }

    return 0;
}

//----------------------------------------------------------------------
vector<int> *BifTetraConns::getCoviseConnections()
{
    return &conn_list;
}

//----------------------------------------------------------------------
vector<int> *BifTetraConns::getCoviseElementList()
{
    if (elem_list.size() == 0)
    {
        for (int i = 0; i < numTetras; i++)
        {
            //elem_list[i] = 4*i;
            elem_list.push_back(4 * i);
        }
    }
    return &elem_list;
}

//----------------------------------------------------------------------
int BifTetraConns::getNumConnections()
{
    return conn_list.size();
}

//----------------------------------------------------------------------
int BifTetraConns::getNumTetras()
{
    return numTetras;
}
