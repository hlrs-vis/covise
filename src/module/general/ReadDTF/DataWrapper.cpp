/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file DataWrapper.cpp
 * @brief contains implementation of class DataWrapper
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 4.12.2003
 * created
 */

#include "DataWrapper.h"

CLASSINFO_OBJ(ClassInfo_DataWrapper, DataWrapper, "DataWrapper", 1);

DataWrapper::DataWrapper()
    : Tools::BaseObject()
{
    INC_OBJ_COUNT(getClassName());
}

DataWrapper::DataWrapper(string className, int objectID)
    : Tools::BaseObject(className, objectID)
{
    INC_OBJ_COUNT(getClassName());
}

DataWrapper::~DataWrapper()
{
    DEC_OBJ_COUNT(getClassName());
}

bool DataWrapper::init()
{
    Tools::ClassManager *cm = Tools::ClassManager::getInstance();

    dataTypes.clear();

    data.clear();

    timeSteps = NULL;

    return true;
}

bool DataWrapper::readData(string fileName,
                           map<int, string> typeList,
                           bool isIndex)
{
    Tools::ClassManager *cm = Tools::ClassManager::getInstance();
    string dataType = "";

    clear();

    dataTypes = typeList;

    if (isIndex)
        return readMultiple(fileName);

    if (!isIndex)
        return readSingle(fileName);

    return false;
}

bool DataWrapper::readSingle(string fileName)
{
    Tools::ClassManager *cm = Tools::ClassManager::getInstance();
    DTF::Data *tsData = NULL;
    bool retVal = false;

    if (fileName != "")
    {
        tsData = (DTF::Data *)cm->getObject("DTF::Data");

        if (fillData(tsData, fileName))
            if (fillTimeSteps())
                retVal = true;
    }

    if (this->data.size() != 0)
        for (int i = 0; i < this->data.size(); i++)
        {
            tsData = this->data[i];
            tsData->clear();
            cm->deleteObject(tsData->getID());
            tsData = NULL;
        }

    this->data.clear();

    return retVal;
}

bool DataWrapper::readMultiple(string indexFile)
{
    Tools::ClassManager *cm = Tools::ClassManager::getInstance();

    vector<string> files;
    DTF::Data *tsData = NULL;

    if (indexFile != "")
    {
        ifstream file(indexFile.c_str());

        if (file)
        {
            char buf[255];
            string line;

            string path = extractPath(indexFile);

            while (file.getline(buf, 255))
            {
                line = path + "/" + buf;
                if (checkForFile(line))
                    files.push_back(line);
            }

            this->data.clear();
            for (int i = 0; i < files.size(); i++)
            {
                tsData = (DTF::Data *)cm->getObject("DTF::Data");

                fillData(tsData, files[i]);
            }

            if (fillTimeSteps())
                return true;
        }
    }

    return false;
}

void DataWrapper::value2Name(int type, string &name)
{
    map<int, string>::iterator i = dataTypes.find(type);

    if (i == dataTypes.end())
        name = dataTypes.find(2)->second;
    else
        name = i->second;
}

void DataWrapper::clear()
{
    Tools::ClassManager *cm = Tools::ClassManager::getInstance();

    if (this->data.size() != 0)
    {
        for (int i = 0; i < this->data.size(); i++)
        {
            DTF::Data *data = this->data[i];
            data->clear();
            cm->deleteObject(data->getID());
            data = NULL;
        }

        this->data.clear();
        this->dataTypes.clear();
    }

    if (this->timeSteps != NULL)
    {
        this->timeSteps->clear();
        cm->deleteObject(this->timeSteps->getID());

        this->timeSteps = NULL;
    }
}

bool DataWrapper::fillData(DTF::Data *tsData, string fileName)
{
    if (tsData != NULL)
    {
        if (tsData->read(fileName))
        {
            this->data.push_back(tsData);
            return true;
        }
        else
            tsData->clear();
    }

    return false;
}

bool DataWrapper::fillTimeSteps()
{
    Tools::ClassManager *cm = Tools::ClassManager::getInstance();

    if (this->timeSteps == NULL)
        this->timeSteps = (DTF::TimeSteps *)cm->getObject("DTF::TimeSteps");

    this->timeSteps->clear();

    if (this->timeSteps->setData(this->data))
        return true;

    return false;
}

coDoSet *DataWrapper::getGridSet(string name, int begin, int end)
{
    coDoSet *gridSet = NULL;
    coDistributedObject **gridObj = NULL;
    int numSteps = 0;
    bool gridError = false;
    char steps[50];
    Tools::Helper *helper = Tools::Helper::getInstance();

    if (this->timeSteps != NULL)
    {
        numSteps = this->timeSteps->getNumTimeSteps();

        if (numSteps > 0)
        {
            if ((begin < 0) || (begin > (numSteps - 1)))
                begin = 0;

            if ((end >= 0) && (end < numSteps - 1))
                numSteps = end;

            gridObj = new coDistributedObject *[numSteps - begin];
            for (int i = 0; i < numSteps - begin; i++)
                gridObj[i] = NULL;

            for (int i = 0 + begin; i < numSteps; i++)
            {
                gridObj[i - begin] = this->getGrid(i);

                if (gridObj[i - begin] == NULL)
                {
                    gridError = true;
                    break;
                }
            }
        }
        else
            gridError = true;
    }
    else
        gridError = true;

    if (!gridError)
    {
        gridSet = new coDoSet(name.c_str(), numSteps - begin, gridObj);
        sprintf(steps, "1_%d", numSteps - begin);
        gridSet->addAttribute("TIMESTEP", steps);
    }

    helper->clearArray(gridObj, numSteps - begin);
    if (gridObj != NULL)
    {
        delete[] gridObj;
        gridObj = NULL;
    }

    return gridSet;
}

coDoSet *DataWrapper::getDataSet(string name, string dataName,
                                 int begin,
                                 int end)
{
    coDoSet *dataSet = NULL;
    coDistributedObject **dataObj = NULL;
    int numSteps = 0;
    bool dataError = false;
    char steps[50];
    Tools::Helper *helper = Tools::Helper::getInstance();

    if (this->timeSteps != NULL)
    {
        numSteps = this->timeSteps->getNumTimeSteps();

        if (numSteps > 0)
        {
            if ((begin < 0) || (begin > (numSteps - 1)))
                begin = 0;

            if ((end >= 0) && (end < numSteps - 1))
                numSteps = end;

            dataObj = new coDistributedObject *[numSteps - begin];

            for (int i = 0 + begin; i < numSteps; i++)
            {
                dataObj[i - begin] = this->getData(i, dataName);

                if (dataObj[i - begin] == NULL)
                {
                    dataError = true;
                    break;
                }
            }
        }
        else
            dataError = true;
    }
    else
        dataError = true;

    if (!dataError)
    {
        dataSet = new coDoSet(name.c_str(), numSteps - begin, dataObj);
        sprintf(steps, "1_%d", numSteps - begin);
        dataSet->addAttribute("TIMESTEP", steps);
    }

    helper->clearArray(dataObj, numSteps - begin);
    if (dataObj != NULL)
    {
        delete[] dataObj;
        dataObj = NULL;
    }

    return dataSet;
}

coDistributedObject *DataWrapper::getGrid(int timeStepNr)
{
    DTF::TimeStep *timeStep = NULL;
    coDistributedObject *grid = NULL;
    char gridName[50];

    if (this->timeSteps != NULL)
    {
        timeStep = this->timeSteps->get(timeStepNr);

        if (timeStep != NULL)
        {
            sprintf(gridName, "grid_%d", timeStepNr);
            grid = timeStep->getGrid(gridName);
        }
    }

    return grid;
}

coDistributedObject *DataWrapper::getData(int timeStepNr, string name)
{
    DTF::TimeStep *timeStep = NULL;
    coDistributedObject *dataObject = NULL;
    char dataName[50];

    if (this->timeSteps != NULL)
    {
        timeStep = this->timeSteps->get(timeStepNr);

        if (timeStep != NULL)
        {
            if (timeStep->hasData())
            {
                sprintf(dataName, "data_%d", timeStepNr);
                dataObject = timeStep->getData(dataName, name);
            }
        }
    }

    return dataObject;
}

bool DataWrapper::hasData()
{
    if (this->timeSteps != NULL)
        if (this->timeSteps->getNumTimeSteps() > 0)
            return true;

    return false;
}

string DataWrapper::extractPath(string fileName)
{
    string path = "";
    string text = fileName;

    const string delim("/");
    vector<string> tokens;

    string::size_type begIndex = text.find_first_not_of(delim);
    string::size_type endIndex;

    while (begIndex != string::npos)
    {
        endIndex = text.find_first_of(delim, begIndex);

        string token(text, begIndex, endIndex - begIndex);

        tokens.push_back(token);

        if (endIndex == string::npos)
            break;

        begIndex = text.find_first_not_of(delim, endIndex + 1);
    }

    for (int i = 0; i < tokens.size() - 1; i++)
        path += "/" + tokens[i];

    return path;
}

bool DataWrapper::checkForFile(string fileName)
{
    ifstream file(fileName.c_str());

    if (file)
        return true;

    return false;
}
