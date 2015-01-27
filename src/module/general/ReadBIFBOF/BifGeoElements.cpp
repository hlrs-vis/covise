/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// IMPLEMENTATION   BifGeoElements
//
// Description: Class for BifBof geometrical connections
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

#include "BifGeoElements.h"

using namespace std;

map<int, vector<int> > BifGeoElements::geoIDs;
vector<int> BifGeoElements::conn_list;
vector<int> BifGeoElements::elem_list;
vector<int> BifGeoElements::type_list;
vector<int> BifGeoElements::poly_list;
vector<int> BifGeoElements::corner_list;

int BifGeoElements::num2dVert = 0;
int BifGeoElements::num3dVert = 0;
// COVISE Constants
static const int COVISE_HEXAGON = 7;
static const int COVISE_HEXAEDER = 7;
static const int COVISE_PRISM = 6;
static const int COVISE_PYRAMID = 5;
static const int COVISE_TETRAHEDER = 4;
static const int COVISE_QUAD = 3;
static const int COVISE_TRIANGLE = 2;
static const int COVISE_BAR = 1;
static const int COVISE_NONE = 0;
static const int COVISE_POINT = 10;

//----------------------------------------------------------------------
//  KONSTUKTOR
//----------------------------------------------------------------------
BifGeoElements::BifGeoElements(int pDsID, int numRec, BifBof *pbifBof)
    : BifElement(pDsID)
    , dsID(pDsID)
    , bifBof(pbifBof)
{
    typeID = this->getTypId();
    numGeos = numRec;

    //make elementList and typeList
    if (typeID == 1) //2d data
    {
        for (int i = 0; i < numGeos; i++)
        {
            poly_list.push_back(num2dVert);
            num2dVert += geoIDs[dsID].at(0);
        }
    }
    if (typeID == 2) //2d data
    {
        for (int i = 0; i < numGeos; i++)
        {
            elem_list.push_back(num3dVert);
            num3dVert += geoIDs[dsID].at(0);
            type_list.push_back(geoIDs[dsID].at(1));
        }
    }
}

//----------------------------------------------------------------------
int BifGeoElements::readInConnections(BifNodalPoints *nodPoints, int &readingComplete)
{
    // get the connectivities of the record buffer
    for (int i = 0; i < numGeos; i++)
    {
        int ret = bifBof->readRegularRecord(elementBuffer, readingComplete);

        if (ret != 0)
            return ret;

        // Transfer data from the entry buffer to the local data structure
        if (typeID == 1)
        {
            for (int j = 2; j < geoIDs[dsID].at(0) + 2; j++)
            {
                corner_list.push_back(nodPoints->getCoviceIdOfBifId(elementBuffer[geoIDs[dsID].at(j)].i));
            }
        }
        if (typeID == 2)
        {
            for (int j = 2; j < geoIDs[dsID].at(0) + 2; j++)
            {
                conn_list.push_back(nodPoints->getCoviceIdOfBifId(elementBuffer[geoIDs[dsID].at(j)].i));
            }
        }
        if (readingComplete != 0)
            return 0;
    }

    return 0;
}

//----------------------------------------------------------------------
vector<int> *BifGeoElements::getCoviseConnections()
{
    return &conn_list;
}

//----------------------------------------------------------------------
vector<int> *BifGeoElements::getCoviseElementList()
{
    return &elem_list;
}

//----------------------------------------------------------------------
vector<int> *BifGeoElements::getCoviseTypeList()
{
    return &type_list;
}
//----------------------------------------------------------------------
vector<int> *BifGeoElements::getCovisePolyList()
{
    return &poly_list;
}
//----------------------------------------------------------------------
vector<int> *BifGeoElements::getCoviseCornerList()
{
    return &corner_list;
}
//----------------------------------------------------------------------
int BifGeoElements::getNumConnections()
{
    return conn_list.size();
}

//----------------------------------------------------------------------
int BifGeoElements::getNumElements()
{
    return elem_list.size();
} //----------------------------------------------------------------------
int BifGeoElements::getNumPolys()
{
    return poly_list.size();
}
//----------------------------------------------------------------------
int BifGeoElements::getNumCorners()
{
    return corner_list.size();
} //----------------------------------------------------------------------
int BifGeoElements::getNumTypes()
{
    return type_list.size();
}
//----------------------------------------------------------------------
int BifGeoElements::getNumGeos()
{
    return numGeos;
}
//----------------------------------------------------------------------
void BifGeoElements::clear()
{
    conn_list.clear();
    elem_list.clear();
    type_list.clear();
    poly_list.clear();
    corner_list.clear();
    num2dVert = 0;
    num3dVert = 0;
}
//----------------------------------------------------------------------
void BifGeoElements::makeGeoIDs()
{
    vector<int> geoVector;

    //Triangular elements
    geoVector.push_back(3); //numVertices
    geoVector.push_back(COVISE_TRIANGLE); //covise_type
    geoVector.push_back(1);
    geoVector.push_back(3);
    geoVector.push_back(2);
    geoVector.push_back(0);
    geoVector.push_back(0);
    geoVector.push_back(0);
    geoVector.push_back(0);
    geoVector.push_back(0);
    geoIDs[BifElement::TRIANGULAR] = geoVector;
    geoIDs[BifElement::TRIANGULAR_CUVED] = geoVector;
    geoIDs[BifElement::TRIANGULAR_VAR] = geoVector;
    geoIDs[BifElement::TRIANGULAR_6] = geoVector;
    geoIDs[BifElement::TRIANGULAR_7] = geoVector;
    geoIDs[BifElement::TRIANGULAR_6_VAR] = geoVector;
    geoIDs[BifElement::TRIANGULAR_9_CUVED] = geoVector;
    //Quadrilateral elements
    geoVector.at(0) = 4; //numVertices
    geoVector.at(1) = COVISE_QUAD; //covise_type
    geoVector.at(2) = 1;
    geoVector.at(3) = 4;
    geoVector.at(4) = 3;
    geoVector.at(5) = 2;
    geoIDs[BifElement::QUADRILATERAL] = geoVector;
    geoIDs[BifElement::QUADRILATERAL_CUVED] = geoVector;
    geoIDs[BifElement::QUADRILATERAL_VAR] = geoVector;
    geoIDs[BifElement::QUADRILATERAL_8_VAR] = geoVector;
    geoIDs[BifElement::QUADRILATERAL_12_CUVED] = geoVector;
    geoIDs[BifElement::QUADRILATERAL_8_CUVED] = geoVector;
    geoIDs[BifElement::QUADRILATERAL_9_CUVED] = geoVector;
    //Tetrahedron elements
    geoVector.at(0) = 4; //numVertices
    geoVector.at(1) = COVISE_TETRAHEDER; //covise_type
    geoVector.at(2) = 1;
    geoVector.at(3) = 2;
    geoVector.at(4) = 3;
    geoVector.at(5) = 4;
    geoIDs[BifElement::TETRAHEDRON] = geoVector;
    geoIDs[BifElement::TETRAHEDRON_10] = geoVector;
    geoIDs[BifElement::TETRAHEDRON_16] = geoVector;
    //Hexahedron elements
    geoVector.at(0) = 8; //numVertices
    geoVector.at(1) = COVISE_HEXAEDER; //covise_type
    geoVector.at(2) = 5;
    geoVector.at(3) = 8;
    geoVector.at(4) = 7;
    geoVector.at(5) = 6;
    geoVector.at(6) = 1;
    geoVector.at(7) = 4;
    geoVector.at(8) = 3;
    geoVector.at(9) = 2;
    geoIDs[BifElement::HEXAHEDRON] = geoVector;
    geoIDs[BifElement::HEXAHEDRON_20] = geoVector;
    geoIDs[BifElement::HEXAHEDRON_21] = geoVector;
    geoIDs[BifElement::HEXAHEDRON_27] = geoVector;
    geoIDs[BifElement::HEXAHEDRON_32] = geoVector;
    //Pyramid elements
    geoVector.at(0) = 5; //numVertices
    geoVector.at(1) = COVISE_PYRAMID; //covise_type
    geoVector.at(2) = 1;
    geoVector.at(3) = 2;
    geoVector.at(4) = 3;
    geoVector.at(5) = 4;
    geoVector.at(6) = 5;
    geoIDs[BifElement::PYRAMID_13] = geoVector;
    geoIDs[BifElement::PYRAMID] = geoVector;
    geoIDs[BifElement::PYRAMID_14] = geoVector;
    //Pentahedron elements
    geoVector.at(0) = 6; //numVertices
    geoVector.at(1) = COVISE_PRISM; //covise_type
    geoVector.at(2) = 6;
    geoVector.at(3) = 5;
    geoVector.at(4) = 4;
    geoVector.at(5) = 3;
    geoVector.at(6) = 2;
    geoVector.at(7) = 1;
    geoIDs[BifElement::PENTAHEDRON] = geoVector;
    geoIDs[BifElement::PENTAHEDRON_15] = geoVector;
    geoIDs[BifElement::PENTAHEDRON_18] = geoVector;
    geoIDs[BifElement::PENTAHEDRON_24] = geoVector;
}
