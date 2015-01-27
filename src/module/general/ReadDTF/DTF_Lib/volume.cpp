/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** BOD */

/** @file DTF_Lib/volume.cpp
 * @brief contains definition of methods for class DTF_Lib::Volume
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 01.10.2003
 * created
 */

/** EOD */

/** BOC */

#include "volume.h"

using namespace DTF_Lib;

CLASSINFO_OBJ(ClassInfo_DTFLibVolume, Volume, "DTF_Lib::Volume", 1);

Volume::Volume()
    : LibObject(){};

Volume::Volume(string className, int objectID)
    : LibObject(className, objectID)
{
    INC_OBJ_COUNT(getClassName());
}

Volume::~Volume()
{
    clear();
    DEC_OBJ_COUNT(getClassName());
}

bool Volume::queryCondition(int simNum,
                            int zoneNum,
                            int conditionNum,
                            int &groupNum,
                            int &recordNum)
{
    return implementMe();
}

bool Volume::queryNumConditions(int simNum,
                                int zoneNum,
                                int &numConditions)
{
    return implementMe();
}

/** EOC */
