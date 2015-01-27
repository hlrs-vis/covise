/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "interfacematlabcom.hxx"
#include <stdlib.h>
#include <sstream>

InterfaceMatlabCom::InterfaceMatlabCom(void)
    : _matlab(0)
{
}

InterfaceMatlabCom::~InterfaceMatlabCom()
{
    if (_matlab != 0)
        CloseConnection();
}

bool InterfaceMatlabCom::Connect(void)
{
    CoInitialize(NULL);
    CLSID clsid;
    CLSIDFromProgID(CA2CT("Matlab.Application"), &clsid);
    _matlab = new CDIMLApp();
    COleException pError;
    return _matlab->CreateDispatch(clsid, &pError);
}

bool InterfaceMatlabCom::IsConnected(void) const
{
    bool retVal = false;
    if (_matlab)
        retVal = true;
    return retVal;
}

void InterfaceMatlabCom::CloseConnection(void)
{
    if (_matlab != NULL)
    {
        delete _matlab;
        _matlab = NULL;
    }
    CoUninitialize();
}

void InterfaceMatlabCom::CloseMatlab(void)
{
    _matlab->Quit();
    CloseConnection();
}

bool InterfaceMatlabCom::IsVisible(void) const
{
    bool retVal = false;
    if (_matlab->get_Visible() == 1)
        retVal = true;
    return retVal;
}

void InterfaceMatlabCom::SetVisible(const bool visible)
{
    if (visible)
        _matlab->put_Visible(1);
    else
        _matlab->put_Visible(0);
}

void InterfaceMatlabCom::Execute(const char *command) const
{
    CString outputMatlab = _matlab->Execute(CA2CT(command));
}

int InterfaceMatlabCom::GetInteger(const char *variableName) const
{
    VARIANT outputMatlab = _matlab->GetVariable(CA2CT(variableName), CA2CT("base"));
    return (int)outputMatlab.dblVal;
}

int *InterfaceMatlabCom::GetIntegerMatrix(const char *variableName, unsigned long &noRows, unsigned long &noColumns) const
{
    VARIANT outputMatlab = _matlab->GetVariable(CA2CT(variableName), CA2CT("base"));
    SAFEARRAY *outputMatlabArray = outputMatlab.parray;
    noRows = outputMatlabArray->rgsabound[1].cElements;
    noColumns = outputMatlabArray->rgsabound[0].cElements;
    int *retVal = new int[noRows * noColumns];
    int *values = (int *)outputMatlabArray->pvData;
    for (unsigned long i = 0; i < noColumns; i++)
    {
        for (unsigned long j = 0; j < noRows; j++)
            retVal[i * noRows + j] = values[i * noRows + j];
    }
    return retVal;
}

double *InterfaceMatlabCom::GetDoubleMatrix(const char *variableName, unsigned long &noRows, unsigned long &noColumns) const
{
    VARIANT outputMatlab = _matlab->GetVariable(CA2CT(variableName), CA2CT("base"));
    SAFEARRAY *outputMatlabArray = outputMatlab.parray;
    noRows = outputMatlabArray->rgsabound[1].cElements;
    noColumns = outputMatlabArray->rgsabound[0].cElements;
    const double *values = (double *)outputMatlabArray->pvData;
    double *retVal = new double[noRows * noColumns];
    for (unsigned long i = 0; i < noColumns; i++)
    {
        for (unsigned long j = 0; j < noRows; j++)
            retVal[i * noRows + j] = values[i * noRows + j]; // Fortran format
    }
    return retVal;
}

void InterfaceMatlabCom::SetInteger(const char *variableName, const int value) const
{
    VARIANT inputMatlab;
    inputMatlab.vt = VT_I4;
    inputMatlab.intVal = value;
    _matlab->PutWorkspaceData(CA2CT(variableName), CA2CT("base"), inputMatlab);
}

void InterfaceMatlabCom::SetDoubleMatrix(const char *variableName, const double values[], const unsigned long noColumns, const unsigned long noRows) const
{
    VARIANT inputMatlab;
    inputMatlab.vt = VT_R8 | VT_ARRAY;
    UINT cDims = 2;
    SAFEARRAYBOUND rgsabound[2];
    rgsabound[0].lLbound = 0;
    rgsabound[0].cElements = noRows;
    rgsabound[1].lLbound = 0;
    rgsabound[1].cElements = noColumns;
    inputMatlab.parray = SafeArrayCreate(VT_R8, cDims, rgsabound);
    double *inputValues = (double *)inputMatlab.parray->pvData;
    for (unsigned long i = 0; i < noColumns; i++)
    {
        for (unsigned long j = 0; j < noRows; j++)
            inputValues[i * noRows + j] = values[i * noRows + j];
    }
    _matlab->PutWorkspaceData(CA2CT(variableName), CA2CT("base"), inputMatlab);
    SafeArrayDestroy(inputMatlab.parray);
}

void InterfaceMatlabCom::SetIntegerMatrix(const char *variableName, const int values[], const unsigned long noColumns, const unsigned long noRows) const
{
    VARIANT inputMatlab;
    inputMatlab.vt = VT_I4 | VT_ARRAY;
    UINT cDims = 2;
    SAFEARRAYBOUND rgsabound[2];
    rgsabound[0].lLbound = 0;
    rgsabound[0].cElements = noRows;
    rgsabound[1].lLbound = 0;
    rgsabound[1].cElements = noColumns;
    inputMatlab.parray = SafeArrayCreate(VT_I4, cDims, rgsabound);
    int *inputValues = (int *)inputMatlab.parray->pvData;
    for (unsigned long i = 0; i < noColumns; i++)
    {
        for (unsigned long j = 0; j < noRows; j++)
            inputValues[i * noRows + j] = values[i * noRows + j];
    }
    _matlab->PutWorkspaceData(CA2CT(variableName), CA2CT("base"), inputMatlab);
    SafeArrayDestroy(inputMatlab.parray);
}

const std::string InterfaceMatlabCom::GetString(const char *variableName) const
{
    const CString outputMatlab = _matlab->GetCharArray(CA2CT(variableName), CA2CT("base"));
    const std::string retVal = CT2A(outputMatlab);
    return retVal;
}

void InterfaceMatlabCom::WriteString(const char *string, const char *variableName) const
{
    _matlab->PutCharArray(CA2CT(variableName), CA2CT("base"), CA2CT(string));
}

bool InterfaceMatlabCom::IsVariable(const char *variableName) const
{
    std::ostringstream temp;
    temp << "checkVariable = " << variableName;
    std::string inputMatlab = temp.str();
    Execute(inputMatlab.c_str());
    Execute("variableExists = exist('checkVariable')");
    const bool retVal = GetInteger("variableExists") == 1;
    Execute("clear checkVariable");
    Execute("clear variableExists");
    return retVal;
}
