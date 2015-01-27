/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// IMPLEMENTATION   BifNodalPoints
//
// Description: Class for the coordinates in nodal points
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
#ifndef BIF_NODAL_POINTS_H
#define BIF_NODAL_POINTS_H

// ----- C++ Header
#include <iostream> //cout
#include <map>
#include <string>

#include "BifBofInterface.h"

#include "BifElement.h"

using namespace std;

class BifNodalPoints : public BifElement
{

public:
    BifNodalPoints(int dsID, int _minID, int _maxID, int numCoords, BifBof *bb);

    // ----- FUNKTIONEN
    int readInCoordinates(int &readingComplete);
    int getCoviseCoordinates(float **x_covise_coords, float **y_covise_coords, float **z_covise_coords);
    int getNumCoordinates();
    int getMinCoordID()
    {
        return minID;
    };
    int getMaxCoordID()
    {
        return maxID;
    };
    int getCoviceIdOfBifId(int covID);

private:
    // ----- VARIABLEN
    int numCoord;
    int minID, maxID;
    int *fileInputUnit;
    float *xcoords;
    float *ycoords;
    float *zcoords;
    map<int, int> bifToCovID;
    BifBof *bifBof;
    BifBof::Word elementBuffer[10000];

    // ----- implizite Definitionen abklemmen
    const BifNodalPoints &operator=(const BifNodalPoints &); //Zuweisungsoperator
    //Copy Konstruktor wird gebraucht
};

#endif
