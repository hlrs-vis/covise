/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** BOD */

/** @file DTF_Lib/bcrecords.cpp
 * @brief contains implementation of methods for class DTF_Lib::BcRecords.
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 01.10.2003
 * created
 */

/** EOD */

/** BOC */

#include "bcrecords.h"

using namespace DTF_Lib;

/** @brief registration at class manager
 *
 * \b Description:
 *
 * The use of this define macro registers the class DTF_Lib::BcRecords at
 * the class manager with following arguments:
 *
 * - ClassInfo_DTFLibBcRecords: name of the class derived from ClassInfo which
 * is responsible for the creation of new objects of this class.
 * - BcRecords: class which is to register at class manager
 * - "DTF_Lib::BcRecords": string which identifies the class
 * - 1: number of maximum objects (1 means there is a single object allowed)
 */
CLASSINFO_OBJ(ClassInfo_DTFLibBcRecords, BcRecords, "DTF_Lib::BcRecords", 1);

BcRecords::BcRecords()
    : LibObject()
{
    INC_OBJ_COUNT(getClassName());
};

BcRecords::BcRecords(string className, int objectID)
    : LibObject(className, objectID)
{
    INC_OBJ_COUNT(getClassName());
}

BcRecords::~BcRecords()
{
    clear();
    DEC_OBJ_COUNT(getClassName());
}

bool BcRecords::queryCategory(int simNum,
                              int zoneNum,
                              int bcNum,
                              int catNum,
                              string &name,
                              string &value)
{
    return implementMe();
}

bool BcRecords::queryCategoryVal(int simNum,
                                 int zoneNum,
                                 int bcNum,
                                 string name,
                                 string &value)
{
    return implementMe();
}

bool BcRecords::queryEvalData(int simNum,
                              int zoneNum,
                              int bcNum,
                              string valueName,
                              int &numInts,
                              int &numReals,
                              int &numStrings)
{
    return implementMe();
}

bool BcRecords::queryEvalMethod(int simNum,
                                int zoneNum,
                                int bcNum,
                                string name,
                                string &evalMethod)
{
    return implementMe();
}

bool BcRecords::queryNumRecords(int simNum,
                                int zoneNum,
                                int &numRecords)
{
    return implementMe();
}

bool BcRecords::queryRecord(int simNum,
                            int zoneNum,
                            int bcNum,
                            int &key,
                            string &type,
                            string &name,
                            int &numCat,
                            int &numVals)
{
    return implementMe();
}

bool BcRecords::queryValName(int simNum,
                             int zoneNum,
                             int bcNum,
                             int valNum,
                             string &name)
{
    return implementMe();
}

bool BcRecords::readEvalData(int simNum,
                             int zoneNum,
                             int bcNum,
                             string name,
                             EvalData &evalData)
{
    return implementMe();
}

bool BcRecords::readVal(int simNum,
                        int zoneNum,
                        int bcNum,
                        string name,
                        string intName,
                        int &value)
{
    return implementMe();
}

bool BcRecords::readVal(int simNum,
                        int zoneNum,
                        int bcNum,
                        string name,
                        string realName,
                        double &value)
{
    return implementMe();
}

bool BcRecords::readVal(int simNum,
                        int zoneNum,
                        int bcNum,
                        string name,
                        string stringName,
                        string &value)
{
    return implementMe();
}

/** EOC */
