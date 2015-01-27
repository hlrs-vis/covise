/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** BOD */

/** @file DTF_Lib/vcrecords.cpp
 * @brief contains definition of methods for class DTF_Lib::VcRecords
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 01.10.2003
 * created
 */

/** EOD */

/** BOC */

#include "vcrecords.h"

using namespace DTF_Lib;

CLASSINFO_OBJ(ClassInfo_DTFLibVcRecords, VcRecords, "DTF_Lib::VcRecords", 1);

VcRecords::VcRecords()
    : LibObject(){};

VcRecords::VcRecords(string className, int objectID)
    : LibObject(className, objectID)
{
    INC_OBJ_COUNT(getClassName());
}

VcRecords::~VcRecords()
{
    clear();
    DEC_OBJ_COUNT(getClassName());
}

bool VcRecords::queryNumRecords(int simNum,
                                int zoneNum,
                                int &numRecords)
{
    return implementMe();
}

bool VcRecords::queryRecord(int simNum,
                            int zoneNum,
                            int vcNum,
                            string &category,
                            string &name,
                            int &numValues)
{
    return implementMe();
}

bool VcRecords::queryEvalData(int simNum,
                              int zoneNum,
                              int vcNum,
                              string name,
                              int &numInts,
                              int &numReals,
                              int &numStrings)
{
    return implementMe();
}

bool VcRecords::queryEvalMethod(int simNum,
                                int zoneNum,
                                int vcNum,
                                string name,
                                string &method)
{
    return implementMe();
}

bool VcRecords::queryValName(int simNum,
                             int zoneNum,
                             int vcNum,
                             int valNum,
                             string &name)
{
    return implementMe();
}

bool VcRecords::readEvalData(int simNum,
                             int zoneNum,
                             int vcNum,
                             string valName,
                             EvalData &evalData)
{
    return implementMe();
}

bool VcRecords::readVal(int simNum,
                        int zoneNum,
                        int vcNum,
                        string name,
                        string intName,
                        int &value)
{
    return implementMe();
}

bool VcRecords::readVal(int simNum,
                        int zoneNum,
                        int vcNum,
                        string name,
                        string realName,
                        double &value)
{
    return implementMe();
}

bool VcRecords::readVal(int simNum,
                        int zoneNum,
                        int vcNum,
                        string name,
                        string stringName,
                        string &value)
{
    return implementMe();
}

/** EOC */
