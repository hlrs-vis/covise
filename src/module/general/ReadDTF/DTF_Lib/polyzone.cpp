/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** BOD */

/** @file DTF_Lib/polyzone.cpp
 * @brief contains definition of methods for class DTF_Lib::PolyZone
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 01.10.2003
 * created
 */

/** EOD */

/** BOC */

#include "polyzone.h"

using namespace DTF_Lib;

CLASSINFO_OBJ(ClassInfo_DTFLibPolyZone, PolyZone, "DTF_Lib::PolyZone", 1);

PolyZone::PolyZone()
    : LibObject(){};

PolyZone::PolyZone(string className, int objectID)
    : LibObject(className, objectID)
{
    INC_OBJ_COUNT(getClassName());
}

PolyZone::~PolyZone()
{
    clear();
    DEC_OBJ_COUNT(getClassName());
}

bool PolyZone::queryIsSorted(int simNum,
                             int zoneNum,
                             bool &result)
{
    return implementMe();
}

bool PolyZone::querySizes(int simNum,
                          int zoneNum,
                          PolyZoneData &pzData)
{
    return implementMe();
}

/** EOC */
