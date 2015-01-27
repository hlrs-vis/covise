/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// IMPLEMENTATION   BifTriangConns
//
// Description: Class for triangular connections
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

#include "BifTriangConns.h"

using namespace std;

//----------------------------------------------------------------------
//  KONSTUKTOR
//----------------------------------------------------------------------
//BifTriangConns::BifTriangConns( int pMinNodeID, int pMinID, int pMaxID, BifBof *bb ): BifElement("INFE3", 31),  minID(pMinID), maxID(pMaxID), bifBof(bb)
BifTriangConns::BifTriangConns(int pMinNodeID, int pNumTrias, BifBof *bb)
    : BifElement(31)
    , bifBof(bb)
{

    //  numTriang = pMaxID-pMinID+1;
    numTriang = pNumTrias;
    numConns = pNumTrias * 3;

    elem_list = new int[numTriang];
    conn_list = new int[numConns];
    minNodeID = pMinNodeID;
}

//----------------------------------------------------------------------
int BifTriangConns::readInConnections(BifNodalPoints *nodPoints, int &readingComplete)
{

    // get the connectivities of the record buffer
    for (int i = 0; i < numTriang; i++)
    {
        int ret = bifBof->readRegularRecord(elementBuffer, readingComplete);

        if (ret != 0)
            return ret;

        /* Transfer data from the entry buffer */
        /* to the local data structure */

        conn_list[3 * i] = nodPoints->getCoviceIdOfBifId(elementBuffer[1].i);
        conn_list[3 * i + 1] = nodPoints->getCoviceIdOfBifId(elementBuffer[2].i);
        conn_list[3 * i + 2] = nodPoints->getCoviceIdOfBifId(elementBuffer[3].i);

        if (readingComplete != 0)
            return 0;
    }
    return 0;
}

//----------------------------------------------------------------------
int *BifTriangConns::getCoviseConnections()
{
    return conn_list;
}

//----------------------------------------------------------------------
void BifTriangConns::getCoviseElementList(int **return_elem_list)
{
    for (int i = 0; i < numTriang; i++)
    {
        elem_list[i] = 3 * i;
    }
    *return_elem_list = elem_list;
}

//----------------------------------------------------------------------
int BifTriangConns::getNumConnections()
{
    return numConns;
}

//----------------------------------------------------------------------
int BifTriangConns::getNumTriangles()
{
    return numTriang;
}
