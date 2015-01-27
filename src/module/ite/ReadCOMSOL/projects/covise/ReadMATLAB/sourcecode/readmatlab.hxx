// ITE-Toolbox ReadMATLAB
//
// (C) Institute for Theory of Electrical Engineering
//
// COVISE module, which reads data from COMSOL
//
// main program

#pragma once

#include <api/coModule.h>
#include "../../../interfaces/matlab/include/interfacematlab.hxx"
#include "datacontainer.hxx"

using namespace covise;

// main class
class ReadMATLAB : public coModule
{
public:
    // constructor
    ReadMATLAB(int argc, char* argv[]);
    // destructor
    virtual ~ReadMATLAB();
protected:
private:
    // COVISE interface functions
    virtual int compute(const char* port);
    virtual void postInst(void);
    virtual void param(const char* paramName, bool inMapLoading);
    // member variables
    InterfaceMatlab* _interfaceMatlab; // interface to MATLAB
    coInputPort* _inGrid;
    coInputPort* _inData1;
    coInputPort* _inData2;
    coOutputPort* _outGrid;
    coOutputPort* _outData;
    coBooleanParam* _paramUseData1;
    coStringParam* _paramNameData1;
    coBooleanParam* _paramUseData2;
    coStringParam* _paramNameData2;
    coStringParam* _paramComuteFunctionName;
    coStringParam* _paramResultName;
    // functions for data processing
    bool ComputeMesh(void);
    bool ComputeValues(void);
    bool CheckInputValues(void) const;
    coDistributedObject* GetOutputGrid(coDistributedObject* inputGrid, const char* nameOutGrid) const;
    unsigned long GetNoGridPoints(const coDistributedObject* grid) const;
    coDistributedObject* CallComputeMatlab(const unsigned int timeStep, const unsigned long noPoints, coDistributedObject* grid, coDistributedObject* data1, coDistributedObject* data2, const char* nameOut);
    void ComputeMatlab(const unsigned int timeStep, const unsigned long noPoints, const double x[], const double y[], const double z[], DataContainer &data1, DataContainer& data2, DataContainer &results);
};
