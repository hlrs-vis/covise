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
#ifndef BIF_TETRA_CONNS_H
#define BIF_TETRA_CONNS_H

// ----- C++ Header
#include <iostream> //cout
#include <vector>
#include <string>

#include "BifBofInterface.h"

#include "BifElement.h"
#include "BifNodalPoints.h"

using namespace std;

class BifTetraConns : public BifElement
{

public:
    BifTetraConns(int dsID, int pMinNodeID, int pMinID, int pMaxID, BifBof *bb);

    // ----- FUNKTIONEN
    int readInConnections(BifNodalPoints *nodPoints, int &readingComplete);
    int getNumConnections();
    int getNumTetras();
    vector<int> *getCoviseConnections();
    vector<int> *getCoviseElementList();

private:
    // ----- VARIABLEN
    int numTetras;
    //       int           numConns;
    int minID, maxID;
    int minNodeID;
    int *fileInputUnit;
    vector<int> elem_list;
    vector<int> conn_list;

    BifBof *bifBof;
    BifBof::Word elementBuffer[10000];

    // ----- implizite Definitionen abklemmen
    const BifTetraConns &operator=(const BifTetraConns &); //Zuweisungsoperator
    //Copy Konstruktor wird gebraucht
};

#endif
