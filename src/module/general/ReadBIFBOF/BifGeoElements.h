/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// IMPLEMENTATION   BifGeoElements
//
// Description: Class for cubical connections
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
#ifndef BIF_CUBE_Elements_H
#define BIF_CUBE_Elements_H

// ----- C++ Header
#include <iostream> //cout
#include <vector>
#include <string>

#include "BifBofInterface.h"

#include "BifElement.h"
#include "BifNodalPoints.h"

using namespace std;

class BifGeoElements : public BifElement
{

public:
    BifGeoElements(int dsID, int numRec, BifBof *bb);

    // ----- FUNKTIONEN
    int readInConnections(BifNodalPoints *nodPoints, int &readingComplete);
    int getNumGeos();

    // ----- KLASSENFUNKTIONEN
    static vector<int> *getCoviseConnections();
    static vector<int> *getCoviseElementList();
    static vector<int> *getCoviseTypeList();

    static vector<int> *getCovisePolyList();
    static vector<int> *getCoviseCornerList();

    static int getNumConnections();
    static int getNumElements();
    static int getNumTypes();
    static int getNumPolys();
    static int getNumCorners();
    static void clear();
    static int num2dVert;
    static int num3dVert;
    // ----- KLASSENVARIABLE
    //map with BifBofID, numVertices, covise_type, order of vertices
    static map<int, vector<int> > geoIDs;
    // ----- KLASSENFUNKTIONEN
    static void makeGeoIDs();

private:
    // ----- KLASSENVARIABLE
    static vector<int> conn_list;
    static vector<int> elem_list;
    static vector<int> type_list;
    static vector<int> poly_list;
    static vector<int> corner_list;

    // ----- VARIABLEN
    int dsID;
    int typeID;
    int numGeos;
    int minID, maxID;
    int *fileInputUnit;

    BifBof *bifBof;
    BifBof::Word elementBuffer[10000];

    // ----- implizite Definitionen abklemmen
    const BifGeoElements &operator=(const BifGeoElements &); //Zuweisungsoperator
    //Copy Konstruktor wird gebraucht
};

#endif
