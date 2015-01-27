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
#ifndef BIF_TRIANG_CONNS_H
#define BIF_TRIANG_CONNS_H

// ----- C++ Header
#include <iostream> //cout
#include <map>

#include "BifBofInterface.h"

#include "BifElement.h"
#include "BifNodalPoints.h"

using namespace std;

class BifTriangConns : public BifElement
{

public:
    //BifTriangConns ( int pMinNodeID, int pMinID, int pMaxID, BifBof *bb );
    BifTriangConns(int pMinNodeID, int pNumTrias, BifBof *bb);
    // ----- FUNKTIONEN
    int readInConnections(BifNodalPoints *nodPoints, int &readingComplete);
    int getNumConnections();
    int getNumTriangles();
    int *getCoviseConnections();
    void getCoviseElementList(int **return_elem_list);
    //       virtual int getPlace( int &error )=0;
    //       virtual string typ() const =0;

private:
    // ----- VARIABLEN
    int numTriang;
    int numConns;
    int minID, maxID;
    int minNodeID;
    int *fileInputUnit;
    int *elem_list;
    int *conn_list;
    int *covise_conn_list;

    BifBof *bifBof;
    BifBof::Word elementBuffer[10000];

    // ----- implizite Definitionen abklemmen
    const BifTriangConns &operator=(const BifTriangConns &); //Zuweisungsoperator
    //Copy Konstruktor wird gebraucht
};

#endif
