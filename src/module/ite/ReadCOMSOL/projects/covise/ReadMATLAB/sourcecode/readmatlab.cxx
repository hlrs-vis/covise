/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ITE-Toolbox ReadCOMSOL
//
// (C) Institute for Theory of Electrical Engineering
//
// COVISE module, which reads data from COMSOL
//
// main program

#include "readmatlab.hxx"
#include "do/codounstructuredgrid.h"
#include "do/codouniformgrid.h"
#include "do/codointarr.h"
#include "do/codoset.h"
#include "do/cododata.h"

ReadMATLAB::ReadMATLAB(int argc, char *argv[])
    : coModule(argc, argv, "ReadMATLAB")
    , _interfaceMatlab(NULL)
{
    // input ports
    _inGrid = addInputPort("inputGrid", "UniformGrid|RectilinearGrid|StructuredGrid|UnstructuredGrid", "grid");
    _inGrid->setRequired(true);
    _inData1 = addInputPort("inputData1", "Float|Vec3", "first set of data on grid");
    _inData1->setRequired(false);
    _inData2 = addInputPort("inputData2", "Float|Vec3", "second set of data on grid");
    _inData2->setRequired(false);
    // output ports
    _outGrid = addOutputPort("outputGrid", "UniformGrid|RectilinearGrid|StructuredGrid|UnstructuredGrid", "input grid or computed grid");
    _outData = addOutputPort("outputData", "Float|Vec3", "computed data on grid");
    // parameters
    _paramUseData1 = addBooleanParam("use_input_data_1", "use first data set on the grid");
    _paramUseData1->setValue(false);
    _paramNameData1 = addStringParam("name_data_1", "name of first data set in MATLAB");
    _paramNameData1->setValue("data1");
    _paramUseData2 = addBooleanParam("use_input_data_2", "use second data set on the grid");
    _paramUseData2->setValue(false);
    _paramNameData2 = addStringParam("name_data_2", "name of second data set in MATLAB");
    _paramNameData2->setValue("data2");
    _paramComuteFunctionName = addStringParam("compute_function_name", "name of the compute function in MATLAB");
    _paramComuteFunctionName->setValue("compute");
    _paramResultName = addStringParam("result_name", "name of the results in MATLAB");
    _paramResultName->setValue("result");
}

ReadMATLAB::~ReadMATLAB()
{
    if (_interfaceMatlab != 0)
        delete _interfaceMatlab;
}

int ReadMATLAB::compute(const char *port)
{
    (void)port;
    bool success = true;
    success = CheckInputValues();
    if (success)
    {
        success = ComputeMesh();
        if (success)
            success = ComputeValues();
    }
    int retVal = SUCCESS;
    if (!success)
        retVal = FAIL;
    return retVal;
}

void ReadMATLAB::postInst(void)
{
    sendInfo("ITE-Toolbox: ReadMATLAB");
    _interfaceMatlab = InterfaceMatlab::getInstance();
    if (_interfaceMatlab->Connect())
        sendInfo("A connection to MATLAB was successfully established.");
    else
        sendError("A connection to MATLAB could not be established.");
    _paramNameData1->disable();
    _paramNameData2->disable();
}

void ReadMATLAB::param(const char *paramName, bool inMapLoading)
{
    (void)inMapLoading;
    if (strcmp(paramName, _paramUseData1->getName()) == 0)
    {
        if (_paramUseData1->getValue())
            _paramNameData1->enable();
        else
            _paramNameData1->disable();
    }
    if (strcmp(paramName, _paramUseData2->getName()) == 0)
    {
        if (_paramUseData2->getValue())
            _paramNameData2->enable();
        else
            _paramNameData2->disable();
    }
}

bool ReadMATLAB::ComputeMesh(void)
{
    bool retVal = true;
    coDistributedObject *inputGrid = _inGrid->getCurrentObject();
    if (coDoSet *inputSet = dynamic_cast<coDoSet *>(inputGrid))
    {
        const int noTimeSteps = inputSet->getNumElements();
        coDistributedObject **allGrids = new coDistributedObject *[noTimeSteps + 1];
        allGrids[noTimeSteps] = NULL;
        const char *outputName = _outGrid->getObjName();
        for (int i = 0; i < noTimeSteps; i++)
        {
            std::stringstream temp;
            temp << outputName << "_" << i << "_grid";
            std::string buf = temp.str();
            allGrids[i] = GetOutputGrid(inputSet->getElement(i), buf.c_str());
        }
        coObjInfo objInfo(_outGrid->getObjName());
        coDoSet *outputGrids = new coDoSet(objInfo, allGrids);
        outputGrids->addAttribute("TIMESTEP", "1 10");
        _outGrid->setCurrentObject(outputGrids);
        delete[] allGrids;
    }
    else
        _outGrid->setCurrentObject(GetOutputGrid(inputGrid, _outGrid->getObjName()));
    return retVal;
}

bool ReadMATLAB::ComputeValues(void)
{
    bool retVal = CheckInputValues();
    if (retVal)
    {
        coDistributedObject *grid = _outGrid->getCurrentObject();
        if (coDoSet *gridSet = dynamic_cast<coDoSet *>(grid))
        {
            sendError("multiple time steps are not supported");
        }
        else
        {
            const unsigned long noPoints = GetNoGridPoints(grid);
            coDistributedObject *result = CallComputeMatlab(0, noPoints, grid, _inData1->getCurrentObject(), _inData2->getCurrentObject(), _outData->getObjName());
            _outData->setCurrentObject(result);
        }
    }
    return retVal;
}

bool ReadMATLAB::CheckInputValues(void) const
{
    bool retVal = true;
    if (_paramUseData1->getValue())
    {
        if (_inData1->getCurrentObject() == NULL)
        {
            sendError("first data set must be given at the corresponding input port");
            retVal = false;
        }
    }
    if (_paramUseData2->getValue())
    {
        if (_inData2->getCurrentObject() == NULL)
        {
            sendError("second data set must be given at the corresponding input port");
            retVal = false;
        }
    }
    return retVal;
}

coDistributedObject *ReadMATLAB::GetOutputGrid(coDistributedObject *inputGrid, const char *nameOutGrid) const
{
    coDistributedObject *retVal = NULL;
    coObjInfo objInfo(nameOutGrid);
    if (coDoUniformGrid *uniformGrid = dynamic_cast<coDoUniformGrid *>(inputGrid))
    {
        int noX, noY, noZ;
        uniformGrid->getGridSize(&noX, &noY, &noZ);
        float minX, maxX, minY, maxY, minZ, maxZ;
        uniformGrid->getMinMax(&minX, &maxX, &minY, &maxY, &minZ, &maxZ);
        retVal = new coDoUniformGrid(objInfo, noX, noY, noZ, minX, maxX, minY, maxY, minZ, maxZ);
    }
    else
    {
        if (coDoUnstructuredGrid *unstructuredGrid = dynamic_cast<coDoUnstructuredGrid *>(inputGrid))
        {
            int noElements, noConnectivity, noCoordinates;
            unstructuredGrid->getGridSize(&noElements, &noConnectivity, &noCoordinates);
            int *elements;
            int *connectivity;
            float *x;
            float *y;
            float *z;
            unstructuredGrid->getAddresses(&elements, &connectivity, &x, &y, &z);
            int *typeList;
            unstructuredGrid->getTypeList(&typeList);
            retVal = new coDoUnstructuredGrid(objInfo, noElements, noConnectivity, noCoordinates, elements, connectivity, x, y, z, typeList);
        }
        else
            sendError("input grid is not implemented");
    }
    return retVal;
}

unsigned long ReadMATLAB::GetNoGridPoints(const coDistributedObject *grid) const
{
    unsigned long retVal = 0;
    if (const coDoUniformGrid *uniformGrid = dynamic_cast<const coDoUniformGrid *>(grid))
    {
        int noX = 0;
        int noY = 0;
        int noZ = 0;
        uniformGrid->getGridSize(&noX, &noY, &noZ);
        retVal = noX * noY * noZ;
    }
    else
        sendError("input grid is not supported");
    return retVal;
}

coDistributedObject *ReadMATLAB::CallComputeMatlab(const unsigned int timeStep, const unsigned long noPoints, coDistributedObject *grid, coDistributedObject *data1, coDistributedObject *data2, const char *nameOut)
{
    double *x = new double[noPoints];
    double *y = new double[noPoints];
    double *z = new double[noPoints];
    if (coDoUniformGrid *uniformGrid = dynamic_cast<coDoUniformGrid *>(grid))
    {
        int noX = 0;
        int noY = 0;
        int noZ = 0;
        uniformGrid->getGridSize(&noX, &noY, &noZ);
        long index = 0;
        for (int i = 0; i < noX; i++)
        {
            for (int j = 0; j < noY; j++)
            {
                for (int k = 0; k < noZ; k++)
                {
                    if (index >= noPoints)
                        sendError("error in noPoints");
                    else
                    {
                        float tempX = 0;
                        float tempY = 0;
                        float tempZ = 0;
                        uniformGrid->getPointCoordinates(i, &tempX, j, &tempY, k, &tempZ);
                        x[index] = tempX;
                        y[index] = tempY;
                        z[index] = tempZ;
                    }
                    index++;
                }
            }
        }
    }
    else
        sendError("input grid is not supported");
    DataContainer data1C(noPoints);
    if (data1 != NULL)
    {
        if (coDoFloat *scalarData = dynamic_cast<coDoFloat *>(data1))
        {
            data1C.SetType(false);
            float *data = scalarData->getAddress();
            double *dataC = data1C.GetX();
            for (unsigned long i = 0; i < noPoints; i++)
                dataC[i] = data[i];
        }
        else
        {
            if (coDoVec3 *vectorData = dynamic_cast<coDoVec3 *>(data1))
            {
                data1C.SetType(true);
                float *dataX;
                float *dataY;
                float *dataZ;
                vectorData->getAddresses(&dataX, &dataY, &dataZ);
                double *dataCX = data1C.GetX();
                double *dataCY = data1C.GetY();
                double *dataCZ = data1C.GetZ();
                for (unsigned long i = 0; i < noPoints; i++)
                {
                    dataCX[i] = dataX[i];
                    dataCY[i] = dataY[i];
                    dataCZ[i] = dataZ[i];
                }
            }
            else
                sendError("data type at port data1 is not supported");
        }
    }
    DataContainer data2C(noPoints);
    if (data2 != NULL)
    {
        if (coDoFloat *scalarData = dynamic_cast<coDoFloat *>(data2))
        {
            data1C.SetType(false);
            float *data = scalarData->getAddress();
            double *dataC = data2C.GetX();
            for (unsigned long i = 0; i < noPoints; i++)
                dataC[i] = data[i];
        }
        else
        {
            if (coDoVec3 *vectorData = dynamic_cast<coDoVec3 *>(data2))
            {
                data2C.SetType(true);
                float *dataX;
                float *dataY;
                float *dataZ;
                vectorData->getAddresses(&dataX, &dataY, &dataZ);
                double *dataCX = data2C.GetX();
                double *dataCY = data2C.GetY();
                double *dataCZ = data2C.GetZ();
                for (unsigned long i = 0; i < noPoints; i++)
                {
                    dataCX[i] = dataX[i];
                    dataCY[i] = dataY[i];
                    dataCZ[i] = dataZ[i];
                }
            }
            else
                sendError("data type at port data2 is not supported");
        }
    }
    DataContainer result(noPoints);
    ComputeMatlab(timeStep, noPoints, x, y, z, data1C, data2C, result);
    coDistributedObject *retVal = NULL;
    coObjInfo objInfo(nameOut);
    if (result.IsVector())
    {
        coDoVec3 *resultCovise = new coDoVec3(objInfo, noPoints);
        const double *resMatlabX = result.GetX();
        const double *resMatlabY = result.GetY();
        const double *resMatlabZ = result.GetZ();
        float *dataResultCoviseX;
        float *dataResultCoviseY;
        float *dataResultCoviseZ;
        resultCovise->getAddresses(&dataResultCoviseX, &dataResultCoviseY, &dataResultCoviseZ);
        for (unsigned long i = 0; i < noPoints; i++)
        {
            dataResultCoviseX[i] = resMatlabX[i];
            dataResultCoviseY[i] = resMatlabY[i];
            dataResultCoviseZ[i] = resMatlabZ[i];
        }
        retVal = resultCovise;
    }
    else
    {
        coDoFloat *resultCovise = new coDoFloat(objInfo, noPoints);
        const double *resMatlab = result.GetX();
        float *dataResultCovise = resultCovise->getAddress();
        for (unsigned long i = 0; i < noPoints; i++)
            dataResultCovise[i] = resMatlab[i];
        retVal = resultCovise;
    }
    delete[] x;
    delete[] y;
    delete[] z;
    return retVal;
}

void ReadMATLAB::ComputeMatlab(const unsigned int timeStep, const unsigned long noPoints, const double x[], const double y[], const double z[], DataContainer &data1, DataContainer &data2, DataContainer &results)
{
    const std::string computeFunctionNameMatlab(_paramComuteFunctionName->getValue());
    const std::string resultNameMatlab(_paramResultName->getValue());
    const std::string data1NameMatlab(_paramNameData1->getValue());
    const std::string data2NameMatlab(_paramNameData2->getValue());
    _interfaceMatlab->SetInteger("noPoints", noPoints);
    _interfaceMatlab->SetDoubleMatrix("x", x, noPoints, 1);
    _interfaceMatlab->SetDoubleMatrix("y", y, noPoints, 1);
    _interfaceMatlab->SetDoubleMatrix("z", z, noPoints, 1);
    std::stringstream constructFunctionName;
    constructFunctionName << resultNameMatlab << " = " << computeFunctionNameMatlab << "(" << timeStep + 1 << ", " << noPoints << ", x, y, z";
    if (data1.IsSet())
    {
        if (data1.IsVector())
        {
            double *temp = new double[noPoints * 3];
            const double *data1x = data1.GetX();
            const double *data1y = data1.GetY();
            const double *data1z = data1.GetZ();
            for (unsigned long i = 0; i < noPoints; i++)
            {
                temp[i * 3 + 0] = data1x[i];
                temp[i * 3 + 1] = data1y[i];
                temp[i * 3 + 2] = data1z[i];
            }
            _interfaceMatlab->SetDoubleMatrix(data1NameMatlab.c_str(), temp, noPoints, 3);
            delete[] temp;
        }
        else
            _interfaceMatlab->SetDoubleMatrix(data1NameMatlab.c_str(), data1.GetX(), noPoints, 1);
        constructFunctionName << ", " << data1NameMatlab;
    }
    if (data2.IsSet())
    {
        if (data2.IsVector())
        {
            double *temp = new double[noPoints * 3];
            const double *data2x = data2.GetX();
            const double *data2y = data2.GetY();
            const double *data2z = data2.GetZ();
            for (unsigned long i = 0; i < noPoints; i++)
            {
                temp[i * 3 + 0] = data2x[i];
                temp[i * 3 + 1] = data2y[i];
                temp[i * 3 + 2] = data2z[i];
            }
            _interfaceMatlab->SetDoubleMatrix(data2NameMatlab.c_str(), temp, noPoints, 3);
            delete[] temp;
        }
        else
            _interfaceMatlab->SetDoubleMatrix(data2NameMatlab.c_str(), data2.GetX(), noPoints, 1);
        constructFunctionName << ", " << data2NameMatlab;
    }
    constructFunctionName << ")";
    const string functionCall(constructFunctionName.str());
    _interfaceMatlab->Execute(functionCall.c_str());
    unsigned long noRows = 0;
    unsigned long noColumns = 0;
    double *result = _interfaceMatlab->GetDoubleMatrix(resultNameMatlab.c_str(), noRows, noColumns);
    if (noRows == 1)
    {
        results.SetType(false);
        double *resultX = results.GetX();
        for (unsigned long i = 0; i < noPoints; i++)
            resultX[i] = result[i];
    }
    else
    {
        results.SetType(true);
        double *resultX = results.GetX();
        double *resultY = results.GetY();
        double *resultZ = results.GetZ();
        for (unsigned long i = 0; i < noPoints; i++)
        {
            resultX[i] = result[i * 3 + 0];
            resultY[i] = result[i * 3 + 1];
            resultZ[i] = result[i * 3 + 2];
        }
    }
}

MODULE_MAIN(IO, ReadMATLAB)
