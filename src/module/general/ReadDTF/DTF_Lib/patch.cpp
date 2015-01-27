/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** BOD */

/** @file DTF_Lib/patch.cpp
 * @brief contains definition of methods for class DTF_Lib::Patch
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 01.10.2003
 * created
 */

/** EOD */

/** BOC */

#include "patch.h"

using namespace DTF_Lib;

CLASSINFO_OBJ(ClassInfo_DTFLibPatch, Patch, "DTF_Lib::Patch", 1);

Patch::Patch()
    : LibObject(){};

Patch::Patch(string className, int objectID)
    : LibObject(className, objectID)
{
    INC_OBJ_COUNT(getClassName());
}

Patch::~Patch()
{
    clear();
    DEC_OBJ_COUNT(getClassName());
}

bool Patch::queryNumPatches(int simNum,
                            int zoneNum,
                            int &numPatches)
{
    return implementMe();
}

bool Patch::queryPatch(int simNum,
                       int zoneNum,
                       int patchNum,
                       map<string, vector<int> > &minMax)
{
    return implementMe();
}

bool Patch::readPatch(int simNum,
                      int zoneNum,
                      int patchNum,
                      vector<int> &records)
{
    return implementMe();
}

/** EOC */
