// ITE-Toolbox interface to MATLAB
//
// (C) Institute for Theory of Electrical Engineering
//
// author: A. Buchau
//
// implementation with COM

#pragma once

#include "../include/interfacematlab.hxx"
#ifdef WIN32
#include "../stdafx.h"
#include "../CDIMLApp.h"
#endif
class InterfaceMatlabCom : public InterfaceMatlab
{
public:
    InterfaceMatlabCom(void);
    ~InterfaceMatlabCom();
    bool Connect(void);
    bool IsConnected(void) const;
    void CloseConnection(void);
    void CloseMatlab(void);
    bool IsVisible(void) const;
    void SetVisible(const bool visible);
    void Execute(const char* command) const;
    int GetInteger(const char* variableName) const;
    void SetInteger(const char* variableName, const int value) const;
    int* GetIntegerMatrix(const char* variableName, unsigned long &noRows, unsigned long &noColumns) const;
    double* GetDoubleMatrix(const char* variableName, unsigned long &noRows, unsigned long &noColumns) const;
    void SetDoubleMatrix(const char* variableName, const double values[], const unsigned long noColumns, const unsigned long noRows) const;
    void SetIntegerMatrix(const char* variableName, const int values[], const unsigned long noColumns, const unsigned long noRows) const;
    const std::string GetString(const char* variableName) const;
    void WriteString(const char* string, const char* variableName) const;
    bool IsVariable(const char* variableName) const;
private:
    CDIMLApp* _matlab;
protected:
};
