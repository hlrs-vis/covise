/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** BOD */

/** @file DTF_Lib/zoneinterface.cpp
 * @brief contains definition of methods for class DTF_Lib::ZoneInterface
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 01.10.2003
 * created
 */

/** EOD */

/** BOC */

#include "zoneinterface.h"

using namespace DTF_Lib;

CLASSINFO_OBJ(ClassInfo_DTFLibZoneInterface, ZoneInterface, "DTF_Lib::ZoneInterface", 1);

ZoneInterface::ZoneInterface()
    : LibObject(){};

ZoneInterface::ZoneInterface(string className, int objectID)
    : LibObject(className, objectID)
{
    INC_OBJ_COUNT(getClassName());
}

ZoneInterface::~ZoneInterface()
{
    clear();
    DEC_OBJ_COUNT(getClassName());
}

bool ZoneInterface::queryNumZI(int simNum,
                               int &numZI)
{
    return implementMe();
}

bool ZoneInterface::queryNumZIforZone(int simNum,
                                      int zoneNum,
                                      int &numZI)
{
    return implementMe();
}

bool ZoneInterface::queryZI(int simNum,
                            int ziNum,
                            int &leftZone,
                            int &rightZone,
                            int &numFaces)
{
    return implementMe();
}

bool ZoneInterface::queryZIforZone(int simNum,
                                   int zoneNum,
                                   int &ziNum,
                                   int &leftZone,
                                   int &rightZone,
                                   int &numFaces)
{
    return implementMe();
}

bool ZoneInterface::readZI(int simNum,
                           int ziNum,
                           vector<int> &facenums_l,
                           vector<int> &facenums_r)
{
    return implementMe();
}

bool ZoneInterface::readZIforZone(int simNum,
                                  int zoneNum,
                                  int &ziNum,
                                  vector<int> &facenum_l,
                                  vector<int> &facenum_r)
{
    return implementMe();
}

/** EOC */
