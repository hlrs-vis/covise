/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ITE-Toolbox interface to MATLAB
//
// (C) Institute for Theory of Electrical Engineering
//
// author: A. Buchau
//
// implementation with MATLAB engine

#include "interfacematlabengine.hxx"
#include <stdlib.h>
#include <sstream>

// constructor
InterfaceMatlabEngine::InterfaceMatlabEngine(void)
//: _matlabEngine(NULL)
{
}

// destructor
InterfaceMatlabEngine::~InterfaceMatlabEngine()
{
}

// establish a connection to MATLAB
bool InterfaceMatlabEngine::Connect(void)
{
    bool retVal = true;
    retVal = false;
    //_matlabEngine = engOpen(NULL);
    //if (IsConnected())
    //{
    //    engOutputBuffer(_matlabEngine, _matlabEngineOutput, 256);
    //    if (!IsVariable("fem"))
    //        retVal = false;
    //}
    //else
    //    retVal = false;
    return retVal;
}

// check if a connection to MATLAB is established
bool InterfaceMatlabEngine::IsConnected(void) const
{
    //    return _matlabEngine != NULL;
    return false;
}

// close connection to MATLAB
void InterfaceMatlabEngine::CloseConnection(void)
{
    //engClose(_matlabEngine);
    //_matlabEngine = NULL;
}

// close MATLAB
void InterfaceMatlabEngine::CloseMatlab(void)
{
    //engClose(_matlabEngine);
    //_matlabEngine = NULL;
}

// check if MATLAB window is visible
bool InterfaceMatlabEngine::IsVisible(void) const
{
    bool retVal = false;
    //    engGetVisible(_matlabEngine, &retVal);
    return retVal;
}

// set visibility of MATLAB
void InterfaceMatlabEngine::SetVisible(const bool visible)
{
    //    engSetVisible(_matlabEngine, visible);
}

// execute a command in MATLAB
void InterfaceMatlabEngine::Execute(const char *command) const
{
    //    engEvalString(_matlabEngine, command);
}

// read an integer from MATLAB
int InterfaceMatlabEngine::GetInteger(const char *variableName) const
{
    //mxArray* retValMatlab = engGetVariable(_matlabEngine, variableName);
    //int retVal = (int) mxGetScalar(retValMatlab);
    //mxDestroyArray(retValMatlab);
    //return retVal;
    return 0;
}

// read an integer matrix from MATLAB
int *InterfaceMatlabEngine::GetIntegerMatrix(const char *variableName, unsigned long &noRows, unsigned long &noColumns) const
{
    //mxArray* retValMatlab = engGetVariable(_matlabEngine, variableName);
    //double* valuesRetValMatlab = mxGetPr(retValMatlab);
    //const mwSize* sizeMatlabArray = mxGetDimensions(retValMatlab);
    //noRows = sizeMatlabArray[0]; // not tested
    //noColumns = sizeMatlabArray[1]; // not tested
    //int* retVal = new int [noRows * noColumns];
    //for (unsigned int i = 0; i < noRows * noColumns; i++)
    //    retVal[i] = (int) valuesRetValMatlab[i];
    //mxDestroyArray(retValMatlab);
    //return retVal;
    return 0;
}

// read a double matrix from MATLAB
double *InterfaceMatlabEngine::GetDoubleMatrix(const char *variableName, unsigned long &noRows, unsigned long &noColumns) const
{
    //mxArray* retValMatlab = engGetVariable(_matlabEngine, variableName);
    //double* valuesRetValMatlab = mxGetPr(retValMatlab);
    //const mwSize* sizeMatlabArray = mxGetDimensions(retValMatlab);
    //noRows = sizeMatlabArray[0]; // not tested
    //noColumns = sizeMatlabArray[1]; // not tested
    //double* retVal = new double [noRows * noColumns];
    //for (unsigned int i = 0; i < noRows * noColumns; i++)
    //    retVal[i] = valuesRetValMatlab[i];
    //mxDestroyArray(retValMatlab);
    //return retVal;
    return 0;
}

// write an integer to MATLAB
void InterfaceMatlabEngine::SetInteger(const char *variableName, const int value) const
{
    //mxArray* variable = mxCreateDoubleMatrix(1, 1, mxREAL);
    //double* valueVariable = mxGetPr(variable);
    //*valueVariable = value;
    //engPutVariable(_matlabEngine, variableName, variable);
    //mxDestroyArray(variable);
}

// write a double matrix to MATLAB
void InterfaceMatlabEngine::SetDoubleMatrix(const char *variableName, const double values[], const unsigned long noColumns, const unsigned long noRows) const
{
    //mxArray* matlabValues = mxCreateDoubleMatrix(noColumns, noRows, mxREAL);
    //double* ptrValues = mxGetPr(matlabValues);
    //for (unsigned int i = 0; i < noColumns * noRows; i++)
    //    ptrValues[i] = values[i];
    //engPutVariable(_matlabEngine, variableName, matlabValues);
    //mxDestroyArray(matlabValues);
}

void InterfaceMatlabEngine::SetIntegerMatrix(const char *variableName, const int values[], const unsigned long noColumns, const unsigned long noRows) const
{
}

// read a string from MATLAB
const std::string InterfaceMatlabEngine::GetString(const char *variableName) const
{
    //mxArray* outputMatlab = engGetVariable(_matlabEngine, variableName);
    //std::string retVal = mxArrayToString(outputMatlab);
    //mxDestroyArray(outputMatlab);
    //return retVal;
    return "";
}

// write a string to MATLAB
void InterfaceMatlabEngine::WriteString(const char *string, const char *variableName) const
{
    std::ostringstream matlabInput;
    matlabInput << variableName << " = \"" << string << "\"";
    std::string temp = matlabInput.str();
    Execute(temp.c_str());
}

// check if a variable in MATLAB exists
bool InterfaceMatlabEngine::IsVariable(const char *variableName) const
{
    std::ostringstream temp;
    temp << "varexists = exist('" << variableName << "', 'var')";
    std::string inputMatlab = temp.str();
    Execute(inputMatlab.c_str());
    return GetInteger("varexists") == 1;
}
