/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** BOD */

/** @file DTF_Lib/virtualzone.cpp
 * @brief contains definition of methods for class DTF_Lib::VirtualZone
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 01.10.2003
 * created
 */

/** EOD */

/** BOC */

#include "virtualzone.h"
#include <cstdio>

using namespace DTF_Lib;

CLASSINFO_OBJ(ClassInfo_DTFLibVirtualZone, VirtualZone, "DTF_Lib::VirtualZone", 1);

VirtualZone::VirtualZone()
    : LibObject(){};

VirtualZone::VirtualZone(string className, int objectID)
    : LibObject(className, objectID)
{
    INC_OBJ_COUNT(getClassName());
}

VirtualZone::~VirtualZone()
{
    clear();
    DEC_OBJ_COUNT(getClassName());
}

bool VirtualZone::queryBCrecNum(int simNum,
                                int zoneNum,
                                int bcRecNum,
                                int &vzBcRecNum)
{
    return implementMe();
}

bool VirtualZone::queryNumVZDs(int simNum,
                               int &numVZDs)
{
    int simNumber = simNum;
    int handle = this->fileHandle;

    if ((numVZDs = dtf_query_nvzds(&handle, &simNumber)) != DTF_ERROR)
        return true;

    return false;
}

bool VirtualZone::queryNumVZDsOfTopotype(int simNum,
                                         int topotype,
                                         int &numVZDs)
{
    return implementMe();
}

bool VirtualZone::queryVZDbyName(int simNum,
                                 string name,
                                 DataElement &vzdInfo)
{
    int handle = this->fileHandle;
    int simNumber = simNum;
    int numElements = 0;
    dtf_string units;
    dtf_datatype datatype;
    dtf_topotype topotype;

    if (dtf_query_vzd_by_name(&handle,
                              &simNumber,
                              name.c_str(),
                              &numElements,
                              &datatype,
                              units,
                              &topotype) != DTF_ERROR)
    {
        vzdInfo.setValues(name, numElements, datatype, units, topotype);
        return true;
    }

    return false;
}

bool VirtualZone::queryVZDbyNum(int simNum,
                                int vzdNum,
                                DataElement &vzdInfo)
{
    int handle = this->fileHandle;
    int simNumber = simNum;
    int numElements = 0;
    int vzdNumber = vzdNum;

    dtf_string name, units;
    dtf_datatype datatype;
    dtf_topotype topotype;

    if (dtf_query_vzd_by_num(&handle,
                             &simNumber,
                             &vzdNumber,
                             name,
                             &numElements,
                             &datatype,
                             units,
                             &topotype) != DTF_ERROR)
    {
        vzdInfo.setValues(name, numElements, datatype, units, topotype);
        return true;
    }

    return false;
}

bool VirtualZone::queryVZDNames(int simNum,
                                vector<string> &names)
{
    Tools::ClassManager *cm = Tools::ClassManager::getInstance();

    int numVZDs = 0;
    DataElement *vzdInfo = (DataElement *)cm->getObject("DTF_Lib::DataElement");

    names.clear();

    if (queryNumVZDs(simNum, numVZDs))
    {
        for (int i = 1; i <= numVZDs; i++)
            if (queryVZDbyNum(simNum, i, *vzdInfo))
            {
                names.push_back(vzdInfo->getName());
                vzdInfo->clear();
            }

        cm->deleteObject(vzdInfo->getID());
        return true;
    }

    cm->deleteObject(vzdInfo->getID());
    return false;
}

bool VirtualZone::readVZDbyName(int simNum,
                                string name,
                                vector<int> &dataStore)
{
    Tools::ClassManager *cm = Tools::ClassManager::getInstance();
    DataElement *zdInfo = (DataElement *)cm->getObject("DTF_Lib::DataElement");

    int handle = this->fileHandle;
    int simNumber = simNum;
    dtf_datatype dataType = DTF_INT_DATA;
    int *data = NULL;
    bool retVal = false;

    dataStore.clear();

    if (queryVZDbyName(simNumber, name, *zdInfo))
        if (zdInfo->getDataType() == dataType)
        {
            data = new int[zdInfo->getNumElements()];

            if (dtf_read_vzd_by_name(&handle,
                                     &simNumber,
                                     name.c_str(),
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

bool VirtualZone::readVZDbyName(int simNum,
                                string name,
                                vector<double> &dataStore)
{
    Tools::ClassManager *cm = Tools::ClassManager::getInstance();
    DataElement *zdInfo = (DataElement *)cm->getObject("DTF_Lib::DataElement");

    int handle = this->fileHandle;
    int simNumber = simNum;
    dtf_datatype dblType = DTF_DOUBLE_DATA;
    dtf_datatype fltType = DTF_SINGLE_DATA;
    double *dblData = NULL;
    float *fltData = NULL;

    bool retVal = false;

    dataStore.clear();

    if (queryVZDbyName(simNumber, name, *zdInfo))
    {
        if ((zdInfo->getDataType() == dblType))
        {
            dblData = new double[zdInfo->getNumElements()];
            if (dtf_read_vzd_by_name(&handle,
                                     &simNumber,
                                     name.c_str(),
                                     dblData,
                                     &dblType) != DTF_ERROR)
            {
                retVal = true;
            }

            if (retVal)
                for (int i = 0; i < zdInfo->getNumElements(); i++)
                    dataStore.push_back(dblData[i]);
        }
        else if (zdInfo->getDataType() == fltType)
        {
            fltData = new float[zdInfo->getNumElements()];

            if (dtf_read_vzd_by_name(&handle,
                                     &simNumber,
                                     name.c_str(),
                                     fltData,
                                     &fltType) != DTF_ERROR)
            {
                retVal = true;
            }

            if (retVal)
                for (int i = 0; i < zdInfo->getNumElements(); i++)
                    dataStore.push_back(fltData[i]);
        }
    }

    if (dblData != NULL)
    {
        delete[] dblData;
        dblData = NULL;
    }

    if (fltData != NULL)
    {
        delete[] fltData;
        fltData = NULL;
    }

    return retVal;
}

bool VirtualZone::readVZDbyName(int simNum,
                                string name,
                                vector<string> &dataStore)
{
    Tools::ClassManager *cm = Tools::ClassManager::getInstance();
    DataElement *zdInfo = (DataElement *)cm->getObject("DTF_Lib::DataElement");

    int handle = this->fileHandle;
    int simNumber = simNum;
    dtf_datatype dataType = DTF_STRING_DATA;
    dtf_string *data = NULL;
    bool retVal = false;

    dataStore.clear();

    if (queryVZDbyName(simNumber, name, *zdInfo))
        if (zdInfo->getDataType() == dataType)
        {
            data = new dtf_string[zdInfo->getNumElements()];

            if (dtf_read_vzd_by_name(&handle,
                                     &simNumber,
                                     name.c_str(),
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

bool VirtualZone::readVZDbyNum(int simNum,
                               int vzdNum,
                               vector<void *> &data,
                               int &datatype)
{
    return implementMe();
}

bool VirtualZone::readVZDNumsOfTopotype(int simNum,
                                        int topotype,
                                        vector<int> &vzdNums)
{
    return implementMe();
}

/** EOC */
