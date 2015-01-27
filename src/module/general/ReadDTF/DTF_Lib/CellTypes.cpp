/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file DTF_Lib/CellTypes.cpp
 * @brief contains implementation of methods of class DTF_Lib::CellTypes.
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 09.10.2003
 * created
 */
#include "CellTypes.h"

using namespace DTF_Lib;

CellTypes::CellTypes()
    : Tools::EnumTypes()
{
    elementLength.clear();

    names.insert(pair<int, string>(1, "Triangle"));
    names.insert(pair<int, string>(2, "Quadrilateral"));
    names.insert(pair<int, string>(3, "Tetrahedron"));
    names.insert(pair<int, string>(4, "Pyramid"));
    names.insert(pair<int, string>(5, "Prism"));
    names.insert(pair<int, string>(6, "Hexahedron"));
    names.insert(pair<int, string>(7, "Quadratic-Triangle"));
    names.insert(pair<int, string>(8, "Quadratic-Quad"));
    names.insert(pair<int, string>(9, "Quadratic-Tet"));
    names.insert(pair<int, string>(10, "Quadratic-Hex"));
    names.insert(pair<int, string>(11, "Polycell"));

    elementLength.insert(pair<int, int>(1, 3));
    elementLength.insert(pair<int, int>(2, 4));
    elementLength.insert(pair<int, int>(3, 4));
    elementLength.insert(pair<int, int>(4, 5));
    elementLength.insert(pair<int, int>(5, 5));
    elementLength.insert(pair<int, int>(6, 6));
    elementLength.insert(pair<int, int>(7, 3));
    elementLength.insert(pair<int, int>(8, 4));
    elementLength.insert(pair<int, int>(9, 4));
    elementLength.insert(pair<int, int>(10, 6));
    elementLength.insert(pair<int, int>(11, -1)); // -1 for N
}

CellTypes::~CellTypes()
{
    elementLength.clear();
}
