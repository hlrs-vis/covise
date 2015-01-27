// ITE-Toolbox interface to MATLAB
//
// (C) Institute for Theory of Electrical Engineering
//
// author: A. Buchau

#pragma once
#include <string>
#include <vector>
#include "../sourcecode/dllapi.h"

class API_INTERFACEMATLAB InterfaceMatlab
{
public:
    InterfaceMatlab(void);
    virtual ~InterfaceMatlab();
    static InterfaceMatlab* getInstance();
    virtual bool Connect(void) = 0;
    virtual bool IsConnected(void) const = 0;
    virtual void CloseConnection(void) = 0;
    virtual void CloseMatlab(void) = 0;
    virtual bool IsVisible(void) const = 0;
    virtual void SetVisible(const bool visible) = 0;
    virtual void Execute(const char* command) const = 0;
    virtual int GetInteger(const char* variableName) const = 0;
    virtual void SetInteger(const char* variableName, const int value) const = 0;
    virtual int* GetIntegerMatrix(const char* variableName, unsigned long &noRows, unsigned long &noColumns) const = 0;
    virtual double* GetDoubleMatrix(const char* variableName, unsigned long &noRows, unsigned long &noColumns) const = 0;
    virtual void SetDoubleMatrix(const char* variableName, const double values[], const unsigned long noColumns, const unsigned long noRows) const = 0;
    virtual void SetIntegerMatrix(const char* variableName, const int values[], const unsigned long noColumns, const unsigned long noRows) const = 0;
    virtual const std::string GetString(const char* variableName) const = 0;
    virtual void WriteString(const char* string, const char* variableName) const = 0;
    virtual bool IsVariable(const char* variableName) const = 0;
private:
protected:
};
