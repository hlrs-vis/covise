/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** BOD */

/** @file DTF_Lib/surface.cpp
 * @brief contains definition of methods for class DTF_Lib::Surface
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 01.10.2003
 * created
 */

/** EOD */

/** BOC */

#include "surface.h"

using namespace DTF_Lib;

CLASSINFO_OBJ(ClassInfo_DTFLibSurface, Surface, "DTF_Lib::Surface", 1);

Surface::Surface()
    : LibObject(){};
Surface::Surface(string className, int objectID)
    : LibObject(className, objectID)
{
    INC_OBJ_COUNT(getClassName());
}

Surface::~Surface()
{
    clear();
    DEC_OBJ_COUNT(getClassName());
}

bool Surface::queryNumConditions(int simNum,
                                 int zoneNum,
                                 int &numConditions)
{
    return implementMe();
}

bool Surface::queryCondition(int simNum,
                             int zoneNum,
                             int condNum,
                             int &groupNum,
                             int &recordNum)
{
    return implementMe();
}

bool Surface::queryNumGroups(int simNum,
                             int zoneNum,
                             int &numGroups)
{
    return implementMe();
}

bool Surface::queryGroup(int simNum,
                         int zoneNum,
                         int groupNum,
                         int &key)
{
    return implementMe();
}

bool Surface::readGroup(int simNum,
                        int zoneNum,
                        int groupNum,
                        vector<int> &faces)
{
    return implementMe();
}

bool Surface::queryFaces(int simNum,
                         int zoneNum,
                         vector<int> &numTypes,
                         vector<int> &numKind)
{
    return implementMe();
}

bool Surface::queryNumFaces(int simNum,
                            int zoneNum,
                            int &numFaces)
{
    return implementMe();
}

bool Surface::queryFaceKind(int simNum,
                            int zoneNum,
                            int faceNum,
                            int &facekind)
{
    return implementMe();
}

/** EOC */
