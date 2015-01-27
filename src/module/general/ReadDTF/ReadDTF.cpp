/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ReadDTF.h"

Tools::ClassManager *cm = Tools::ClassManager::getInstance();

ReadDTF::ReadDTF(int argc, char *argv[])
    : coModule(argc, argv, "ReadDTF program")
{
    gridPort = addOutputPort("ugridPort",
                             "coDoUnstructuredGrid",
                             "port for unstructured grids");

    fileParam = addFileBrowserParam("filename", "Path to DTF file");
    fileParam->setValue("./.", "*.DTF");

    isIndexParam = addBooleanParam("file is index", "Indicates if file is index file");
    isIndexParam->setValue(1);

    tsStartParam = addInt32Param("start at timestep", "Starting TimeStep");
    tsStartParam->setValue(0);

    tsEndParam = addInt32Param("stop at timestep", "Stopping TimeStep");
    tsEndParam->setValue(100);

    fillTypeList();
    createDataParams();
    createDataPorts();

    wrapper = NULL;

    updateDataOnly = false;
}

void ReadDTF::postInst()
{
    fileParam->show();
    isIndexParam->show();

    tsStartParam->show();
    tsEndParam->show();

    for (int i = 0; i < dataParams.size(); i++)
    {
        coChoiceParam *dataParam = dataParams[i];
        dataParam->show();
    }
}

void ReadDTF::param(const char *paramName)
{
    string name = paramName;

    if (name == "filename")
    {
        sendInfo("Updating data... this may take a while. Please stay were you are - dont run away");
        selfExec();
    }

    if (name.find("cfd_data") != string::npos)
    {
        sendInfo("Updating data... this may take a while. Please stay were you are - dont run away");
        updateDataOnly = true;
        selfExec();
    }

    if (name.find("timestep") != string::npos)
    {
        sendInfo("Updating data... this may take a while. Please stay were you are - dont run away");
        updateDataOnly = true;
        selfExec();
    }
}

int ReadDTF::compute(const char *)
{
    sendInfo("ReadDTF::compute(const char *)");
    coDoSet *gridSet = NULL;
    coDoSet *dataSet = NULL;

    string fileName = fileParam->getValue();
    bool isIndex = false;

    if (wrapper == NULL)
        wrapper = (DataWrapper *)cm->getObject("DataWrapper");

    if (!updateDataOnly || !wrapper->hasData())
    {
        wrapper->clear();
        if (isIndexParam->getValue())
            isIndex = true;

        if (wrapper->readData(fileName, this->dataTypes, isIndex))
        {
            int begin = this->tsStartParam->getValue();
            int end = this->tsEndParam->getValue();

            gridSet = wrapper->getGridSet(gridPort->getObjName(),
                                          begin,
                                          end);

            if (gridSet != NULL)
                gridPort->setCurrentObject(gridSet);

            updateDataPort();
        }
    }
    else
    {
        gridSet = wrapper->getGridSet(gridPort->getObjName(),
                                      this->tsStartParam->getValue(),
                                      this->tsEndParam->getValue());

        if (gridSet != NULL)
            gridPort->setCurrentObject(gridSet);

        updateDataPort();
    }

    updateDataOnly = false;

    sendInfo("end compute()");

    return SUCCESS;
}

void ReadDTF::fillTypeList()
{
    dataTypes.clear();

    dataTypes.insert(pair<int, string>(1, "None"));
    dataTypes.insert(pair<int, string>(2, "Cell_U"));
    dataTypes.insert(pair<int, string>(3, "Cell_V"));
    dataTypes.insert(pair<int, string>(4, "Cell_W"));
    dataTypes.insert(pair<int, string>(5, "Cell_RHO"));
    dataTypes.insert(pair<int, string>(6, "Cell_P"));
    dataTypes.insert(pair<int, string>(7, "Cell_MU"));
    dataTypes.insert(pair<int, string>(8, "Cell_T"));

    dataTypes.insert(pair<int, string>(9, "RHO"));
    dataTypes.insert(pair<int, string>(10, "U"));
    dataTypes.insert(pair<int, string>(11, "V"));
    dataTypes.insert(pair<int, string>(12, "W"));
    dataTypes.insert(pair<int, string>(13, "P"));
    dataTypes.insert(pair<int, string>(14, "P_tot"));
    dataTypes.insert(pair<int, string>(15, "Vislam"));
    dataTypes.insert(pair<int, string>(16, "T"));
}

void ReadDTF::quit()
{
    Tools::Helper *helper = Tools::Helper::getInstance();

    if (wrapper != NULL)
    {
        wrapper->clear();
        cm->deleteObject(wrapper->getID());
    }

    helper->clearArray(helper->vectorToArray(dataPorts), dataPorts.size());
    helper->clearArray(helper->vectorToArray(dataParams), dataParams.size());

    dataPorts.clear();
    dataParams.clear();

    cout << "deleting cm" << endl;
    delete cm;
}

void ReadDTF::updateDataPort()
{
    if (wrapper != NULL)
        for (int i = 0; i < dataParams.size(); i++)
        {
            coChoiceParam *dataParam = dataParams[i];
            coOutputPort *dataPort = dataPorts[i];
            string dataName = "";
            int begin = tsStartParam->getValue();
            int end = tsEndParam->getValue();

            if (dataParam->getValue() != 0)
            {
                wrapper->value2Name(dataParam->getValue(), dataName);

                coDoSet *dataSet = wrapper->getDataSet(dataPort->getObjName(),
                                                       dataName,
                                                       begin,
                                                       end);

                if (dataSet != NULL)
                    dataPort->setCurrentObject(dataSet);
            }

            dataParam = NULL;
            dataPort = NULL;
        }
}

bool ReadDTF::createDataParams()
{
    char **labels = new char *[dataTypes.size()];
    coChoiceParam *dataParam = NULL;
    char paramName[20];
    Tools::Helper *helper = Tools::Helper::getInstance();

    map<int, string>::iterator it = dataTypes.begin();

    for (int i = 0; i < dataTypes.size(); i++)
    {
        string value = it->second;
        labels[i] = new char[value.length()];
        sprintf(labels[i], "%s", value.c_str());

        ++it;
    }

    for (int i = 0; i < 10; i++)
    {
        sprintf(paramName, "cfd_data_%d", i);
        dataParam = addChoiceParam(paramName, "Type of CFD data to show");
        dataParam->setValue(dataTypes.size(), labels, 0);

        if (dataParam != NULL)
            dataParams.push_back(dataParam);

        dataParam = NULL;
    }

    helper->clearArray(labels, dataTypes.size());
    if (labels != NULL)
    {
        delete[] labels;
        labels = NULL;
    }

    return true;
}

bool ReadDTF::createDataPorts()
{
    coOutputPort *dataPort = NULL;
    char portName[20];

    for (int i = 0; i < 10; i++)
    {
        sprintf(portName, "us3dPort_%d", i);
        dataPort = addOutputPort(portName,
                                 "coDoFloat",
                                 "port for unstructured S3D data");
        if (dataPort != NULL)
            dataPorts.push_back(dataPort);

        dataPort = NULL;
    }

    return true;
}

MODULE_MAIN(, ReadDTF)
