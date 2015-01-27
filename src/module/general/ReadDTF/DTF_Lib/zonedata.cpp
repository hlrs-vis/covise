/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** BOD */

/** @file DTF_Lib/zonedata.cpp
 * @brief contains definition of methods for class DTF_Lib::ZoneData
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 01.10.2003
 * created
 */

/** EOD */

/** BOC */

#include "zonedata.h"

using namespace DTF_Lib;

CLASSINFO_OBJ(ClassInfo_DTFLibZoneData, ZoneData, "DTF_Lib::ZoneData", 1);

ZoneData::ZoneData()
    : LibObject(){};

ZoneData::ZoneData(string className, int objectID)
    : LibObject(className, objectID)
{
    INC_OBJ_COUNT(getClassName());
}

ZoneData::~ZoneData()
{
    clear();
    DEC_OBJ_COUNT(getClassName());
}

bool ZoneData::queryMinMax(int simNum,
                           int zoneNum,
                           vector<double> &minMax)
{
    return implementMe();
}

bool ZoneData::queryNumZDs(int simNum,
                           int zoneNum,
                           int &numZDs)
{
    int handle = this->fileHandle;
    int simNumber = simNum;
    int zoneNumber = zoneNum;

    if ((numZDs = dtf_query_nzds(&handle, &simNumber, &zoneNumber))
        != DTF_ERROR)
        return true;

    return false;
}

bool ZoneData::queryNumZDsOfTopotype(int simNum,
                                     int zoneNum,
                                     int topotype,
                                     int &numZDs)
{
    return implementMe();
}

bool ZoneData::queryZDbyName(int simNum,
                             int zoneNum,
                             string name,
                             DataElement &zdInfo)
{
    int simNumber = simNum;
    int zoneNumber = zoneNum;
    int handle = this->fileHandle;
    int numElements = 0;
    dtf_string units;
    dtf_datatype datatype;
    dtf_topotype topotype;

    if (dtf_query_zd_by_name(&handle,
                             &simNumber,
                             &zoneNumber,
                             name.c_str(),
                             &numElements,
                             &datatype,
                             units,
                             &topotype) != DTF_ERROR)
    {
        zdInfo.setValues(name, numElements, datatype, units, topotype);
        return true;
    }

    return false;
}

bool ZoneData::queryZDbyNum(int simNum,
                            int zoneNum,
                            int dataNum,
                            DataElement &zdInfo)
{
    int simNumber = simNum;
    int zoneNumber = zoneNum;
    int dataNumber = dataNum;
    int handle = this->fileHandle;
    int numElements = 0;
    dtf_string name, units;
    dtf_datatype type;
    dtf_topotype topotype;

    if (dtf_query_zd_by_num(&handle,
                            &simNumber,
                            &zoneNumber,
                            &dataNumber,
                            name,
                            &numElements,
                            &type,
                            units,
                            &topotype))
    {
        zdInfo.setValues(name, numElements, type, units, topotype);
        return true;
    }

    return false;
}

bool ZoneData::queryZDNames(int simNum,
                            int zoneNum,
                            vector<string> &names)
{
    int numZDs = 0;
    Tools::ClassManager *cm = Tools::ClassManager::getInstance();
    DataElement *zdInfo = NULL;

    names.clear();

    if (queryNumZDs(simNum, zoneNum, numZDs))
    {
        zdInfo = (DataElement *)cm->getObject("DTF_Lib::DataElement");
        for (int i = 0; i < numZDs; i++)
            if (queryZDbyNum(simNum, zoneNum, i, *zdInfo))
                names.push_back(zdInfo->getName());

        if (zdInfo != NULL)
        {
            cm->deleteObject(zdInfo->getID());
            zdInfo = NULL;
        }

        return true;
    }

    return false;
}

bool ZoneData::readNumZDsOfTopotype(int simNum,
                                    int zoneNum,
                                    int topotype,
                                    vector<int> &nums)
{
    return implementMe();
}

bool ZoneData::readZDbyName(int simNum,
                            int zoneNum,
                            string name,
                            int elementNum,
                            vector<int> &dataStore)
{
    Tools::ClassManager *cm = Tools::ClassManager::getInstance();
    DataElement *zdInfo = (DataElement *)cm->getObject("DTF::DataElement");

    int handle = this->fileHandle;
    int simNumber = simNum;
    int zoneNumber = zoneNum;
    int elementNumber = elementNum;
    dtf_datatype dataType = DTF_INT_DATA;
    int *data = NULL;
    bool retVal = false;

    dataStore.clear();

    if (queryZDbyName(simNumber, zoneNumber, name, *zdInfo))
        if (zdInfo->getDataType() == dataType)
        {
            data = new int[zdInfo->getNumElements()];

            if (dtf_read_zd_by_name(&handle,
                                    &simNumber,
                                    &zoneNumber,
                                    name.c_str(),
                                    &elementNumber,
                                    data,
                                    &dataType) != DTF_ERROR)
            {
                for (int i = 0; i < zdInfo->getNumElements(); i++)
                    dataStore.push_back(data[i]);

                retVal = true;
            }
        }

    if (data != NULL)
    {
        delete[] data;
        data = NULL;
    }

    return retVal;
}

bool ZoneData::readZDbyName(int simNum,
                            int zoneNum,
                            string name,
                            int elementNum,
                            vector<double> &dataStore)
{
    Tools::ClassManager *cm = Tools::ClassManager::getInstance();
    DataElement *zdInfo = (DataElement *)cm->getObject("DTF::DataElement");

    int handle = this->fileHandle;
    int simNumber = simNum;
    int zoneNumber = zoneNum;
    int elementNumber = elementNum;
    dtf_datatype dataType = DTF_DOUBLE_DATA;
    double *data = NULL;
    bool retVal = false;

    dataStore.clear();

    if (queryZDbyName(simNumber, zoneNumber, name, *zdInfo))
        if (zdInfo->getDataType() == dataType)
        {
            data = new double[zdInfo->getNumElements()];

            if (dtf_read_zd_by_name(&handle,
                                    &simNumber,
                                    &zoneNumber,
                                    name.c_str(),
                                    &elementNumber,
                                    data,
                                    &dataType) != DTF_ERROR)
            {
                for (int i = 0; i < zdInfo->getNumElements(); i++)
                    dataStore.push_back(data[i]);

                retVal = true;
            }
        }

    if (data != NULL)
    {
        delete[] data;
        data = NULL;
    }

    return retVal;
}

bool ZoneData::readZDbyName(int simNum,
                            int zoneNum,
                            string name,
                            int elementNum,
                            vector<string> &dataStore)
{
    Tools::ClassManager *cm = Tools::ClassManager::getInstance();
    DataElement *zdInfo = (DataElement *)cm->getObject("DTF::DataElement");

    int handle = this->fileHandle;
    int simNumber = simNum;
    int zoneNumber = zoneNum;
    int elementNumber = elementNum;
    dtf_datatype dataType = DTF_STRING_DATA;
    dtf_string *data = NULL;
    bool retVal = false;

    dataStore.clear();

    if (queryZDbyName(simNumber, zoneNumber, name, *zdInfo))
        if (zdInfo->getDataType() == dataType)
        {
            data = new dtf_string[zdInfo->getNumElements()];

            if (dtf_read_zd_by_name(&handle,
                                    &simNumber,
                                    &zoneNumber,
                                    name.c_str(),
                                    &elementNumber,
                                    data,
                                    &dataType) != DTF_ERROR)
            {
                for (int i = 0; i < zdInfo->getNumElements(); i++)
                    dataStore.push_back(data[i]);

                retVal = true;
            }
        }

    if (data != NULL)
    {
        delete[] data;
        data = NULL;
    }

    return retVal;
}

bool ZoneData::readZDbyNum(int simNum,
                           int zoneNum,
                           int dataNum,
                           int elementNum,
                           vector<int> &data)
{
    return implementMe();
}

bool ZoneData::readZDbyNum(int simNum,
                           int zoneNum,
                           int dataNum,
                           int elementNum,
                           vector<double> &data)
{
    return implementMe();
}

bool ZoneData::readZDbyNum(int simNum,
                           int zoneNum,
                           int dataNum,
                           int elementNum,
                           vector<string> &data)
{
    return implementMe();
}

/** EOC */
