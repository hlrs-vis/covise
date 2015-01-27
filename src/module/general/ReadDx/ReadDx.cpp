/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                            (C)1999 RUS   **
 **                                                                          **
 ** Description:                                                             **
 **                                                                          **
 **                                                                          **
 **                                                                          **
 ** Name:        ReadDx                                                      **
 ** Category:    examples                                                    **
 **                                                                          **
 ** Author: C. Schwenzer                                                     **
 **                                                                          **
 ** History:                                                                 **
 **                                                                          **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include "ReadDx.h"
#include "DxFile.h"
#include <util/coviseCompat.h>
#include <api/coFeedback.h>
#include <do/coDoData.h>

ReadDx::ReadDx(int argc, char *argv[]) // vvvv --- this info appears in the module setup window
    : coModule(argc, argv, "Read module for IBM data explorer")
{
    // output port
    // parameters:
    //   port name
    //   string to indicate connections, convention: name of the data type
    //   description
    p_UnstrGrid = addOutputPort("grid", "StructuredGrid|UnstructuredGrid|Polygons", "Grid Object");
    int i;
    for (i = 0; i < maxDataPorts; i++)
    {
        char name[1000];
        sprintf(name, "Scalar%c", (char)((int)'A' + i));
        p_ScalarData[i] = addOutputPort(name, "Float", "Data on object");
    }
    const char *choiceVal[] = { "None", "StepFile", "Normal" };
    p_timeStepMode = addChoiceParam("ModeForTimesteps", "Decide whether and how to handle timesteps");
    p_timeStepMode->setValue(3, choiceVal, NONE);

    p_timeSteps = addInt32Param("timesteps", "timesteps");
    p_timeSteps->setValue(1);
    p_skip = addInt32Param("skipped_files", "number of skipped files for each timestep");
    p_skip->setValue(0);

    p_stepPath = addFileBrowserParam("fullpath", "Browser");
    p_stepPath->setValue(".", "*.dx;*.DX/*");

    p_xScale = addFloatSliderParam("ScaleX", "scale grid in X Direction");
    p_xScale->setValue(-100.0, 100.0, 1.0);
    p_yScale = addFloatSliderParam("ScaleY", "scale grid in Y Direction");
    p_yScale->setValue(-100.0, 100.0, 1.0);
    p_zScale = addFloatSliderParam("ScaleZ", "scale grid in Z Direction");
    p_zScale->setValue(-100.0, 100.0, 1.0);

    p_defaultIsLittleEndian = addBooleanParam("DefaultIsLittleEndian", "Default is little endian");
    p_defaultIsLittleEndian->setValue(1);
}

bool ReadDx::makeGridSet(const char *objName, coDistributedObject **d, MultiGridMember *m, DxObjectMap &arrays, DxObjectMap & /*fields*/, const char *dxPath, int /*number*/, int *&reverse, int &reverseSize)
{
    coDoUnstructuredGrid *resultGrid;
    DxObject *o = m->getObject();
    DxObject *positions = arrays[o->getPositions()];
    DxObject *connections = arrays[o->getConnections()];
    DxFile *posFile;
    const char *fileName = positions->getFileName();
    //selfContained means that
    //data are located in the dx File after the keyword "end"
    //and not in a separate file
    bool selfContained = (NULL == fileName);
    if (selfContained)
    {
        fileName = dxPath;
    }
    int nconn = (connections->getShape()) * (connections->getItems());

    int ncoord = positions->getItems();
    //int shape= connections->getShape();
    //for dx files all connections have the same type;
    int nelem = connections->getItems();

    //char *dObjectName=new char[100];
    posFile = new DxFile(fileName, selfContained);
    bool stat_ = posFile->isValid();
    if (!stat_)
    {
        return false;
    }
    reverse = new int[ncoord];
    float *xCoord, *yCoord, *zCoord;
    int *tmpConnections;
    // *tmpElements,  , *tmpTypes;tmpElements=new int[nelem];
    tmpConnections = new int[nconn];

    xCoord = new float[ncoord];
    yCoord = new float[ncoord];
    zCoord = new float[ncoord];
    float xScale = p_xScale->getValue();
    float yScale = p_yScale->getValue();
    float zScale = p_zScale->getValue();
    posFile->readCoords(xCoord, xScale,
                        yCoord, yScale,
                        zCoord, zScale,
                        positions->getDataOffset(),
                        positions->getItems(),
                        positions->getByteOrder());
    int i;
    for (i = 0; i < ncoord; i++)
    {
        reverse[i] = -1;
    }

    //read in the connections
    delete posFile;
    fileName = connections->getFileName();
    selfContained = (NULL == fileName);
    if (selfContained)
    {
        fileName = dxPath;
    }
    DxFile *connFile = NULL;
    connFile = new DxFile(fileName, selfContained);
    stat_ = connFile->isValid();
    if (!stat_)
    {
        return false;
    }

    int offs = connections->getDataOffset();
    int shap = connections->getShape();
    int ite = connections->getItems();
    int bo = connections->getByteOrder();
    connFile->readConnections(tmpConnections, offs, shap, ite, bo);
    delete connFile;
    //Now we compute these positions we _really_ need, i.e. which are mentioned in the
    //connections
    int currPosition = 0;
    for (i = 0; i < (connections->getShape() * connections->getItems()); i++)
    {
        if (-1 == reverse[tmpConnections[i]])
        {
            reverse[tmpConnections[i]] = currPosition++;
        }
        tmpConnections[i] = reverse[tmpConnections[i]];
    }
    reverseSize = currPosition;

    resultGrid = new coDoUnstructuredGrid(objName, nelem, nconn, reverseSize, 1);

    //read in the positions
    float *resultXCoord, *resultYCoord, *resultZCoord;
    int *resultElements, *resultConnections, *resultTypes;

    resultGrid->getAddresses(&resultElements, &resultConnections, &resultXCoord, &resultYCoord, &resultZCoord);
    for (i = 0; i < ncoord; i++)
    {
        if (-1 != reverse[i])
        {
            resultXCoord[reverse[i]] = xCoord[i];
            resultYCoord[reverse[i]] = yCoord[i];
            resultZCoord[reverse[i]] = zCoord[i];
        }
    }
    for (i = 0; i < (connections->getShape() * connections->getItems()); i++)
    {
        resultConnections[i] = tmpConnections[i];
    }
    delete xCoord;
    delete yCoord;
    delete zCoord;
    delete tmpConnections;

    int val = 0;
    resultGrid->getTypeList(&resultTypes);
    int *curElem = resultElements;
    const char *elementType = connections->getElementType();
    int gridType;
    if (0 == strcmp(elementType, "cubes"))
    {
        gridType = TYPE_HEXAEDER;
    }
    else if (0 == strcmp(elementType, "quads"))
    {
        gridType = TYPE_QUAD;
    }
    else if (0 == strcmp(elementType, "triangles"))
    {
        gridType = TYPE_TRIANGLE;
    }
    else
    {
        char msg[1024];
        sprintf(msg, "The element type %s is not yet supported", elementType);
        Covise::sendError(msg);
    }
    for (i = 0; i < nelem; i++)
    {
        *resultTypes++ = gridType;
        *curElem++ = val;
        val += connections->getShape();
    }
    char *partName = new char[1 + strlen(m->getName())];
    strcpy(partName, m->getName());
    resultGrid->addAttribute("PART", partName);
    *d = resultGrid;
    return true;
}

bool ReadDx::makeDataSet(coDistributedObject ***d, MultiGridMember *m, DxObjectMap &arrays, const char *dxPath, int number, int timeStepNo, int &numberOfDataSets, int *reverse, int reverseSize)
{
    DxObject *o = m->getObject();
    if (NULL == o->getData())
    {
        // There are no data at all
        numberOfDataSets = 0;
        return true;
    }
    DxObject *data = arrays[o->getData()];

    const char *fileName = data->getFileName();
    //selfContained means that
    //data are located in the dx File after the keyword "end"
    //and not in a separate file
    bool selfContained = (NULL == fileName);
    if (selfContained)
    {
        fileName = dxPath;
    }
    DxFile *dataFile = new DxFile(fileName, selfContained);
    bool stat_ = dataFile->isValid();
    if (!stat_)
    {
        return false;
    }

    numberOfDataSets = data->getShape();
    //If the rank of the data set
    //is 0, the shape is not given.
    //And must be 1 in the logic of our program
    if (0 == data->getRank())
    {
        numberOfDataSets = 1;
    }
    int sizeOfDataSets = data->getItems();
    int j;
    float *values[maxDataPorts];
    for (j = 0; j < numberOfDataSets; j++)
    {
        values[j] = new float[sizeOfDataSets];
    }
    dataFile->readData(
        values,
        data->getDataOffset(),
        numberOfDataSets,
        sizeOfDataSets,
        data->getByteOrder());
    float *resultValues[maxDataPorts];
    for (j = 0; j < numberOfDataSets; j++)
    {
        char *dObjectName = new char[1024];
        const char *objName = p_ScalarData[j]->getObjName();
        sprintf(dObjectName, "%s_%d_%d", objName, timeStepNo, number);
        coDoFloat *res = new coDoFloat(dObjectName, reverseSize);
        d[j][number] = res;
        int i;
        res->getAddress(&(resultValues[j]));
        for (i = 0; i < sizeOfDataSets; i++)
        {
            if (-1 != reverse[i])
            {
                resultValues[j][reverse[i]] = values[j][i];
            }
        }
    }
    for (j = 0; j < numberOfDataSets; j++)
    {
        delete values[j];
    }
    delete dataFile;
    return true;
}

bool ReadDx::computeTimeStep(const char *fileName, int timeStepNo, coDistributedObject **resultGrid, const char *gridObjName, coDistributedObject ***timeData, int &numberOfDataSets)
{
    int i;
    //First parse the dx-File and collect all
    //informations of the structure
    actionClass *a;
    if (p_defaultIsLittleEndian->getValue())
    {
        a = new actionClass(Parser::LSB);
    }
    else
    {
        a = new actionClass(Parser::MSB);
    }
    Parser *parser = new Parser(a, fileName);
    if (!parser->isOpen())
    {
        char msg[1024];
        sprintf(msg, "file %s could not be opened", fileName);
        Covise::sendError(msg);
        delete a;
        delete parser;
        return false;
    }

    parser->yyparse();
    if (!parser->isCorrect())
    {
        delete a;
        delete parser;
        return false;
    }
    //We get a list of fields ( in terms of data explorer )
    //building a multigrid ( in terms of data explorer )
    MemberList &m = a->getMembers();

    int numParts = m.size();
    coDistributedObject **gridSet = new coDistributedObject *[numParts + 1];
    gridSet[numParts] = NULL;

    //old   MemberList::iterator it;
    int **reverse = new int *[numParts];
    int *reverseSize = new int[numParts];
    char partName[1024];
    //old   for(it=m.begin(),i=0; it!=m.end(); it++,i++) {

    for (i = 0; i < numParts; i++)
    {
        sprintf(partName, "%s_%d", gridObjName, i);
        //old makeGridSet(partName, gridSet+i, (*it),a->getArrays(), a->getFields(),fileName,i, reverse[i], reverseSize[i]);
        bool status = makeGridSet(partName, gridSet + i, m.get(i), a->getArrays(), a->getFields(), fileName, i, reverse[i], reverseSize[i]);
        if (!status)
        {
            delete parser;
            delete a;
            return false;
        }
    }
    *resultGrid = new coDoSet(gridObjName, gridSet);

    char ts[1024];
    sprintf(ts, "1 %d", numParts);
    if ((numParts > 1) && (p_timeStepMode->getValue() == NORMAL))
    {
        char ts[1024];
        sprintf(ts, "1 %d", numParts);
        (*resultGrid)->addAttribute("TIMESTEP", ts);
    }

    //Since the number of datasets, each of which is
    //assigned to one port, is not yet known
    //we prepare all Ports
    coDistributedObject **dataSet[maxDataPorts];
    for (i = 0; i < maxDataPorts; i++)
    {
        dataSet[i] = new coDistributedObject *[numParts + 1];
        dataSet[i][numParts] = NULL;
    }

    //When data has the shape n, then n ports for data are used
    //it is computed in makeDataSet
    int partNumber;

    //old   for(it=m.begin(),partNumber=0,i=0; it!=m.end(); it++,partNumber++,i++) {
    //HERE
    for (i = 0, partNumber = 0; i < numParts; i++, partNumber++)
    {
        bool status = makeDataSet(dataSet, m.get(i), a->getArrays(), fileName, partNumber, timeStepNo, numberOfDataSets, reverse[partNumber], reverseSize[i]);
        delete[] reverse[partNumber];
        if (!status)
        {
            delete a;
            delete parser;
            return false;
        }
    }
    delete[] reverseSize;

    for (i = 0; i < numberOfDataSets; i++)
    {
        const char *name = p_ScalarData[i]->getObjName();
        char objName[1024];
        sprintf(objName, "%s_%d", name, timeStepNo);
        timeData[i][timeStepNo] = new coDoSet(objName, dataSet[i]);
        if ((numParts > 1) && (p_timeStepMode->getValue() == NORMAL))
        {
            timeData[i][timeStepNo]->addAttribute("TIMESTEP", ts);
        }
    }

    delete a;
    delete parser;
    return true;
}

int ReadDx::compute(const char *)
{
    int numberOfTimeSteps = p_timeSteps->getValue();
    int skipValue = p_skip->getValue();
    const char *stepPath = p_stepPath->getValue();
    coStepFile *stepFile = new coStepFile(stepPath);
    stepFile->set_skip_value(skipValue);
    char *nextPath = NULL;

    coDistributedObject **timeGrids = new coDistributedObject *[1 + numberOfTimeSteps];
    timeGrids[numberOfTimeSteps] = NULL;
    coDistributedObject **timeData[maxDataPorts];
    int i;
    for (i = 0; i < maxDataPorts; i++)
    {
        timeData[i] = new coDistributedObject *[1 + numberOfTimeSteps];
        timeData[i][numberOfTimeSteps] = NULL;
    }

    const char *gridName = p_UnstrGrid->getObjName();
    char partName[1024];
    int numberOfDataSets;
    for (i = 0; i < numberOfTimeSteps; i++)
    {
        stepFile->get_nextpath(&nextPath);
        if (NULL != nextPath)
        {
            sprintf(partName, "%s_%d", gridName, i);
            bool status = computeTimeStep(nextPath, i, &(timeGrids[i]), partName, timeData, numberOfDataSets);
            delete[] nextPath;
            if (!status)
            {
                return FAIL;
            }
        }
        else
        {
            Covise::sendError("The indicated number of timesteps is bigger than the number of available files");
            return FAIL;
        }
    }

    coDoSet *gridSet = new coDoSet(gridName, timeGrids);
    char ts[1024];
    sprintf(ts, "1 %d", numberOfTimeSteps);
    if ((numberOfTimeSteps > 1) && (p_timeStepMode->getValue() == STEPFILE))
    {
        gridSet->addAttribute("TIMESTEP", ts);
    }
    p_UnstrGrid->setCurrentObject(gridSet);

    int j;
    coDoSet *dataSet[maxDataPorts];
    for (j = 0; j < numberOfDataSets; j++)
    {
        const char *dataName = p_ScalarData[j]->getObjName();
        dataSet[j] = new coDoSet(dataName, timeData[j]);
        if ((numberOfTimeSteps > 1) && (p_timeStepMode->getValue() == STEPFILE))
        {
            dataSet[j]->addAttribute("TIMESTEP", ts);
        }
        p_ScalarData[j]->setCurrentObject(dataSet[j]);
    }

    return SUCCESS;
}

ReadDx::~ReadDx()
{
}

MODULE_MAIN(Reader, ReadDx)
