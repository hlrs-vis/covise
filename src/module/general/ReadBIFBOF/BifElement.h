/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef BIF_ELEMENT_H
#define BIF_ELEMENT_H

// ----- C++ Header
#include <iostream> //cout
#include <map>

using namespace std;

class BifElement
{

public:
    BifElement(int pId);

    // ----- FUNKTIONEN
    string getName();
    int getTypId();
    int getId();

    // ----- KLASSENVARIABLE
    // Nodal Points
    static const int NODALPOINTS; // "NPCO"
    // Triangular elements
    static const int TRIANGULAR; // "INFE3"
    static const int TRIANGULAR_CUVED; // "INFE3L"
    static const int TRIANGULAR_VAR; // "INFE3V"
    static const int TRIANGULAR_6; // "INFE6"
    static const int TRIANGULAR_7; // "INFE7"
    static const int TRIANGULAR_6_VAR; // "INFE6V"
    static const int TRIANGULAR_9_CUVED; // "INFE9T"
    // Quadrilateral elements
    static const int QUADRILATERAL; // "INFE4"
    static const int QUADRILATERAL_CUVED; // "INFE4L"
    static const int QUADRILATERAL_VAR; // "INFE4V"
    static const int QUADRILATERAL_8_VAR; // "INFE8V"
    static const int QUADRILATERAL_12_CUVED; // "INFE12"
    static const int QUADRILATERAL_8_CUVED; // "INFE8"
    static const int QUADRILATERAL_9_CUVED; // "INFE9"
    // Tetrahedron elements
    static const int TETRAHEDRON; // "INFE4S"
    static const int TETRAHEDRON_10; // "INFE10S"
    static const int TETRAHEDRON_16; // "INFE16S"
    // Pentahedron elements
    static const int PENTAHEDRON; // "INFE6S"
    static const int PENTAHEDRON_15; // "INFE15S"
    static const int PENTAHEDRON_18; // "INFE18S"
    static const int PENTAHEDRON_24; // "INFE24S"
    // Hexahedron elements
    static const int HEXAHEDRON; // "INFE8S"
    static const int HEXAHEDRON_20; // "INFE20S"
    static const int HEXAHEDRON_21; // "INFE21S"
    static const int HEXAHEDRON_27; // "INFE27S"
    static const int HEXAHEDRON_32; // "INFE32S"
    // Pyramid elements
    static const int PYRAMID_13; // "INFE13S"
    static const int PYRAMID; // "INFE5S"
    static const int PYRAMID_14; // "INFE14S"
    // Nodal point temperatures (bof)
    static const int TEMP; // "TEMP"
    //Nodal point deformations
    static const int DEFO; //"DEFO"
    // Definition of parts
    static const int PART; // "PART"
    //---------------------------
    // Nodal Points
    static const int t_NODALPOINTS; // "NPCO"
    // Triangular elements
    static const int t_TRIANGULAR; // "INFE3"
    static const int t_TRIANGULAR_CUVED; // "INFE3L"
    static const int t_TRIANGULAR_VAR; // "INFE3V"
    static const int t_TRIANGULAR_6; // "INFE6"
    static const int t_TRIANGULAR_7; // "INFE7"
    static const int t_TRIANGULAR_6_VAR; // "INFE6V"
    static const int t_TRIANGULAR_9_CUVED; // "INFE9T"
    // Quadrilateral elements
    static const int t_QUADRILATERAL; // "INFE4"
    static const int t_QUADRILATERAL_CUVED; // "INFE4L"
    static const int t_QUADRILATERAL_VAR; // "INFE4V"
    static const int t_QUADRILATERAL_8_VAR; // "INFE8V"
    static const int t_QUADRILATERAL_12_CUVED; // "INFE12"
    static const int t_QUADRILATERAL_8_CUVED; // "INFE8"
    static const int t_QUADRILATERAL_9_CUVED; // "INFE9"
    // Tetrahedron elements
    static const int t_TETRAHEDRON; // "INFE4S"
    static const int t_TETRAHEDRON_10; // "INFE10S"
    static const int t_TETRAHEDRON_16; // "INFE16S"
    // Pentahedron elements
    static const int t_PENTAHEDRON; // "INFE6S"
    static const int t_PENTAHEDRON_15; // "INFE15S"
    static const int t_PENTAHEDRON_18; // "INFE18S"
    static const int t_PENTAHEDRON_24; // "INFE24S"
    // Hexahedron elements
    static const int t_HEXAHEDRON; // "INFE8S"
    static const int t_HEXAHEDRON_20; // "INFE20S"
    static const int t_HEXAHEDRON_21; // "INFE21S"
    static const int t_HEXAHEDRON_27; // "INFE27S"
    static const int t_HEXAHEDRON_32; // "INFE32S"
    // Pyramid elements
    static const int t_PYRAMID_13; // "INFE13S"
    static const int t_PYRAMID; // "INFE5S"
    static const int t_PYRAMID_14; // "INFE14S"
    // Nodal point temperatures (bof)
    static const int t_TEMP; // "TEMP"
    //Nodal point deformations
    static const int t_DEFO; //"DEFO"
    // Definition of parts
    static const int t_PART; // "PART"

private:
    // ----- KLASSENVARIABLE
    static map<int, string> dseleMap;
    static map<int, int> dstypMap;
    // ----- VARIABLEN
    string name;
    int type;
    int id;

    // ----- KLASSENFUNKTION
    static void makeDsele();
    static void makedstyp();

    // ----- implizite Definitionen abklemmen
    const BifElement &operator=(const BifElement &); //Zuweisungsoperator
    //Copy Konstruktor wird gebraucht
};

#endif
