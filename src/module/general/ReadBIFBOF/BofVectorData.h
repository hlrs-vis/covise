/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// IMPLEMENTATION   BofVector Data
//
// Description: Class for the bof scalar temperatur
//              derived from BifElement
//
// Initial version: 12.2008
//
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2008 by HLRS
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
// Changes:
//
#ifndef BOFTENSORDATA_H
#define BOFTENSORDATA_H

// ----- C++ Header
#include <iostream> //cout
#include <string>

#include "BifBofInterface.h"

#include "BifElement.h"
#include "BifNodalPoints.h"

using namespace std;

class BofVectorData : public BifElement
{

public:
    BofVectorData(int dsID, int pCoordMinID, int pCoordMaxID, int pNumCoords, int pMinID, int pMaxID, BifBof *bb);

    // ----- FUNKTIONEN
    int readVectorData(BifNodalPoints *nodPoints, int &readingComplete);
    float *getXArray();
    float *getYArray();
    float *getZArray();
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
    float *xvalue;
    float *yvalue;
    float *zvalue;

    BifBof *bifBof;
    BifBof::Word elementBuffer[10000];

    // ----- implizite Definitionen abklemmen
    const BofVectorData &operator=(const BofVectorData &); //Zuweisungsoperator
    //Copy Konstruktor wird gebraucht
};

#endif
