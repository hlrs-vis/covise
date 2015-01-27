/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// IMPLEMENTATION   BofScalar Data
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
#ifndef BOFSCALARDATA_H
#define BOFSCALARDATA_H

// ----- C++ Header
#include <iostream> //cout
#include <string>

#include "BifBofInterface.h"

#include "BifElement.h"
#include "BifNodalPoints.h"

using namespace std;

class BofScalarData : public BifElement
{

public:
    BofScalarData(int dsID, int pCoordMinID, int pCoordMaxID, int pNumCoords, int pMinID, int pMaxID, BifBof *bb);

    // ----- FUNKTIONEN
    int readScalarData(BifNodalPoints *nodPoints, int &readingComplete);
    float *getScalarArray();
    int getNumCoordinates();
    int getMinCoordID()
    {
        return minID;
    };

private:
    // ----- VARIABLEN
    int numCoord;
    int coordMinID, coordMaxID;
    int minID, maxID;
    int *fileInputUnit;
    float *scalarvalue;

    BifBof *bifBof;
    BifBof::Word elementBuffer[10000];

    // ----- implizite Definitionen abklemmen
    const BofScalarData &operator=(const BofScalarData &); //Zuweisungsoperator
    //Copy Konstruktor wird gebraucht
};

#endif
