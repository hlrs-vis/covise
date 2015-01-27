/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// IMPLEMENTATION   BifElement
//
// Description: Superclass for bif elements
//
// Initial version: 06.2008
//
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2008 by Visenso
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
// Changes:
//

#include "BifElement.h"

#define DSELE_MAP(name) dseleMap[name] = #name;
#define DSTYP_MAP(name) dstypMap[name] = t_##name;

const int BifElement::NODALPOINTS = 1;
const int BifElement::TRIANGULAR = 31;
const int BifElement::TRIANGULAR_CUVED = 32;
const int BifElement::TRIANGULAR_VAR = 34;
const int BifElement::TRIANGULAR_6 = 36;
const int BifElement::TRIANGULAR_7 = 37;
const int BifElement::TRIANGULAR_6_VAR = 38;
const int BifElement::TRIANGULAR_9_CUVED = 39;
const int BifElement::QUADRILATERAL = 41;
const int BifElement::QUADRILATERAL_CUVED = 42;
const int BifElement::QUADRILATERAL_VAR = 44;
const int BifElement::QUADRILATERAL_8_VAR = 46;
const int BifElement::QUADRILATERAL_12_CUVED = 47;
const int BifElement::QUADRILATERAL_8_CUVED = 48;
const int BifElement::QUADRILATERAL_9_CUVED = 49;
const int BifElement::TETRAHEDRON = 61;
const int BifElement::TETRAHEDRON_10 = 62;
const int BifElement::TETRAHEDRON_16 = 73;
const int BifElement::PENTAHEDRON = 63;
const int BifElement::PENTAHEDRON_15 = 64;
const int BifElement::PENTAHEDRON_18 = 69;
const int BifElement::PENTAHEDRON_24 = 74;
const int BifElement::HEXAHEDRON = 65;
const int BifElement::HEXAHEDRON_20 = 66;
const int BifElement::HEXAHEDRON_21 = 67;
const int BifElement::HEXAHEDRON_27 = 68;
const int BifElement::HEXAHEDRON_32 = 75;
const int BifElement::PYRAMID_13 = 70;
const int BifElement::PYRAMID = 71;
const int BifElement::PYRAMID_14 = 72;
const int BifElement::TEMP = 202;
const int BifElement::DEFO = 231;
const int BifElement::PART = 9001;
//Defining Datatypes
//1-2D,2-3D,0-rest
const int BifElement::t_NODALPOINTS = 0;
const int BifElement::t_TRIANGULAR = 1;
const int BifElement::t_TRIANGULAR_CUVED = 1;
const int BifElement::t_TRIANGULAR_VAR = 1;
const int BifElement::t_TRIANGULAR_6 = 1;
const int BifElement::t_TRIANGULAR_7 = 1;
const int BifElement::t_TRIANGULAR_6_VAR = 1;
const int BifElement::t_TRIANGULAR_9_CUVED = 1;
const int BifElement::t_QUADRILATERAL = 1;
const int BifElement::t_QUADRILATERAL_CUVED = 1;
const int BifElement::t_QUADRILATERAL_VAR = 1;
const int BifElement::t_QUADRILATERAL_8_VAR = 1;
const int BifElement::t_QUADRILATERAL_12_CUVED = 1;
const int BifElement::t_QUADRILATERAL_8_CUVED = 1;
const int BifElement::t_QUADRILATERAL_9_CUVED = 1;
const int BifElement::t_TETRAHEDRON = 2;
const int BifElement::t_TETRAHEDRON_10 = 2;
const int BifElement::t_TETRAHEDRON_16 = 2;
const int BifElement::t_PENTAHEDRON = 2;
const int BifElement::t_PENTAHEDRON_15 = 2;
const int BifElement::t_PENTAHEDRON_18 = 2;
const int BifElement::t_PENTAHEDRON_24 = 2;
const int BifElement::t_HEXAHEDRON = 2;
const int BifElement::t_HEXAHEDRON_20 = 2;
const int BifElement::t_HEXAHEDRON_21 = 2;
const int BifElement::t_HEXAHEDRON_27 = 2;
const int BifElement::t_HEXAHEDRON_32 = 2;
const int BifElement::t_PYRAMID_13 = 2;
const int BifElement::t_PYRAMID = 2;
const int BifElement::t_PYRAMID_14 = 2;
const int BifElement::t_TEMP = 0;
const int BifElement::t_DEFO = 0;
const int BifElement::t_PART = 0;

map<int, string> BifElement::dseleMap;
map<int, int> BifElement::dstypMap;
//----------------------------------------------------------------------
//  KONSTUKTOR
//----------------------------------------------------------------------
BifElement::BifElement(int pId)
{
    if (dseleMap.empty())
    {
        makeDsele();
        makedstyp();
    }

    id = pId;
    name = dseleMap[id];
    type = dstypMap[id];
}

//----------------------------------------------------------------------
string BifElement::getName()
{
    return name;
}
int BifElement::getTypId()
{
    return type;
}
//----------------------------------------------------------------------NODALPOINTS
int BifElement::getId()
{
    return (int)id;
}

//----------------------------------------------------------------------
void BifElement::makeDsele()
{
    // Nodal Points
    DSELE_MAP(NODALPOINTS);
    // Triangular elements
    DSELE_MAP(TRIANGULAR);
    DSELE_MAP(TRIANGULAR_CUVED);
    DSELE_MAP(TRIANGULAR_VAR);
    DSELE_MAP(TRIANGULAR_6);
    DSELE_MAP(TRIANGULAR_7);
    DSELE_MAP(TRIANGULAR_6_VAR);
    DSELE_MAP(TRIANGULAR_9_CUVED);
    // Quadrilateral elements
    DSELE_MAP(QUADRILATERAL);
    DSELE_MAP(QUADRILATERAL_CUVED);
    DSELE_MAP(QUADRILATERAL_VAR);
    DSELE_MAP(QUADRILATERAL_8_VAR);
    DSELE_MAP(QUADRILATERAL_12_CUVED);
    DSELE_MAP(QUADRILATERAL_8_CUVED);
    DSELE_MAP(QUADRILATERAL_9_CUVED);
    // Tetrahedron elements
    DSELE_MAP(TETRAHEDRON);
    DSELE_MAP(TETRAHEDRON_10);
    DSELE_MAP(TETRAHEDRON_16);
    // Pentahedron elements
    DSELE_MAP(PENTAHEDRON);
    DSELE_MAP(PENTAHEDRON_15);
    DSELE_MAP(PENTAHEDRON_18);
    DSELE_MAP(PENTAHEDRON_24);
    // Hexahedron elements
    DSELE_MAP(HEXAHEDRON);
    DSELE_MAP(HEXAHEDRON_20);
    DSELE_MAP(HEXAHEDRON_21);
    DSELE_MAP(HEXAHEDRON_27);
    DSELE_MAP(HEXAHEDRON_32);
    // Pyramid elements
    DSELE_MAP(PYRAMID_13);
    DSELE_MAP(PYRAMID);
    DSELE_MAP(PYRAMID_14);
    // Nodal point temperatures (bof)
    DSELE_MAP(TEMP);
    //Nodal point deformations
    DSELE_MAP(DEFO); //"DEFO"
    // Definition of parts
    DSELE_MAP(PART);

    //map<int,string>::iterator it;
}
//----------------------------------------------------------------------
void BifElement::makedstyp()
{
    // Nodal Points
    DSTYP_MAP(NODALPOINTS);
    // Triangular elements
    DSTYP_MAP(TRIANGULAR);
    DSTYP_MAP(TRIANGULAR_CUVED);
    DSTYP_MAP(TRIANGULAR_VAR);
    DSTYP_MAP(TRIANGULAR_6);
    DSTYP_MAP(TRIANGULAR_7);
    DSTYP_MAP(TRIANGULAR_6_VAR);
    DSTYP_MAP(TRIANGULAR_9_CUVED);
    // Quadrilateral elements
    DSTYP_MAP(QUADRILATERAL);
    DSTYP_MAP(QUADRILATERAL_CUVED);
    DSTYP_MAP(QUADRILATERAL_VAR);
    DSTYP_MAP(QUADRILATERAL_8_VAR);
    DSTYP_MAP(QUADRILATERAL_12_CUVED);
    DSTYP_MAP(QUADRILATERAL_8_CUVED);
    DSTYP_MAP(QUADRILATERAL_9_CUVED);
    // Tetrahedron elements
    DSTYP_MAP(TETRAHEDRON);
    DSTYP_MAP(TETRAHEDRON_10);
    DSTYP_MAP(TETRAHEDRON_16);
    // Pentahedron elements
    DSTYP_MAP(PENTAHEDRON);
    DSTYP_MAP(PENTAHEDRON_15);
    DSTYP_MAP(PENTAHEDRON_18);
    DSTYP_MAP(PENTAHEDRON_24);
    // Hexahedron elements
    DSTYP_MAP(HEXAHEDRON);
    DSTYP_MAP(HEXAHEDRON_20);
    DSTYP_MAP(HEXAHEDRON_21);
    DSTYP_MAP(HEXAHEDRON_27);
    DSTYP_MAP(HEXAHEDRON_32);
    // Pyramid elements
    DSTYP_MAP(PYRAMID_13);
    DSTYP_MAP(PYRAMID);
    DSTYP_MAP(PYRAMID_14);
    // Nodal point temperatures (bof)
    DSTYP_MAP(TEMP);
    //Nodal point deformations
    DSELE_MAP(DEFO); //"DEFO"
    // Definition of parts
    DSTYP_MAP(PART);

    //map<int,string>::iterator it;
}
