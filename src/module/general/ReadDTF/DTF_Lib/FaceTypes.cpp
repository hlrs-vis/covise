/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file DTF_Lib/FaceTypes.cpp
 * @brief Contains implementation of methods of class DTF_Lib::CellTypes
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 09.10.2003
 * created
 */

#include "FaceTypes.h"

using namespace DTF_Lib;

FaceTypes::FaceTypes()
    : Tools::EnumTypes()
{
    elementLength.clear();

    names.insert(pair<int, string>(1, "Edge"));
    names.insert(pair<int, string>(2, "Triangle"));
    names.insert(pair<int, string>(3, "Quadrilateral"));
    names.insert(pair<int, string>(4, "Quadratic-Edge"));
    names.insert(pair<int, string>(5, "Quadratic-Triangle"));
    names.insert(pair<int, string>(6, "Quadratic-Quad"));
    names.insert(pair<int, string>(7, "Polyface"));

    elementLength.insert(pair<int, int>(1, 2));
    elementLength.insert(pair<int, int>(2, 3));
    elementLength.insert(pair<int, int>(3, 4));
    elementLength.insert(pair<int, int>(4, 3));
    elementLength.insert(pair<int, int>(5, 6));
    elementLength.insert(pair<int, int>(6, 8));
    elementLength.insert(pair<int, int>(7, -1)); // -1 for N
}

FaceTypes::~FaceTypes()
{
    elementLength.clear();
}
