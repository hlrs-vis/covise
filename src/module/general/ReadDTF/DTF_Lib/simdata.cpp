/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** BOD */

/** @file DTF_Lib/simdata.cpp
 * @brief contains definition of methods for class DTF_Lib::SimData
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 01.10.2003
 * created
 */

/** EOD */

/** BOC */

#include "simdata.h"

using namespace DTF_Lib;

CLASSINFO_OBJ(ClassInfo_DTFLibSimData, SimData, "DTF_Lib::SimData", 1);

SimData::SimData()
    : LibObject(){};

SimData::SimData(string className, int objectID)
    : LibObject(className, objectID)
{
    INC_OBJ_COUNT(getClassName());
}

SimData::~SimData()
{
    clear();
    DEC_OBJ_COUNT(getClassName());
}

bool SimData::queryNumSDsOfTopotype(int simNum,
                                    int topotype,
                                    int &numSDs)
{
    return implementMe();
}

bool SimData::querySDbyName(int simNum,
                            string name,
                            DataElement &sdInfo)
{
    return implementMe();
}

bool SimData::querySDbyNum(int simNum,
                           int dataNum,
                           DataElement &sdInfo)
{
    dtf_string name;
    dtf_int numElements;
    dtf_datatype datatype;
    dtf_string units;
    dtf_topotype topotype;
    int status;
    int handle = this->fileHandle;
    int simNumber = simNum;
    int dataNumber = dataNum;

    if ((status = dtf_query_sd_by_num(&handle,
                                      &simNumber,
                                      &dataNumber,
                                      name,
                                      &numElements,
                                      &datatype,
                                      units,
                                      &topotype)) != DTF_ERROR)
    {
        string dataName = name;
        string dataUnits = units;

        sdInfo.setValues(dataName, numElements, datatype, dataUnits,
                         topotype);

        return true;
    }

    return false;
}

bool SimData::readNumSDsOfTopotype(int simNum,
                                   int topotype,
                                   vector<int> &nums)
{
    return implementMe();
}

bool SimData::readSDbyName(int simNum,
                           string name,
                           int elementNum,
                           int &datatype,
                           vector<void *> &data)
{
    return implementMe();
}

bool SimData::readSDbyNum(int simNum,
                          int dataNum,
                          int elementNum,
                          int &datatype,
                          vector<void *> &data)
{
    return implementMe();
}

/** EOC */
