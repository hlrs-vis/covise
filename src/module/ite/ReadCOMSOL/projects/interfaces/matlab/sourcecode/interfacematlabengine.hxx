// ITE-Toolbox interface to MATLAB
//
// (C) Institute for Theory of Electrical Engineering
//
// author: A. Buchau
//
// implementation with MATLAB engine

#pragma once

#include "../include/interfacematlab.hxx"
//#include "engine.h"

class InterfaceMatlabEngine : public InterfaceMatlab
{
public:
    // constructor
    InterfaceMatlabEngine(void);
    // destructor
    ~InterfaceMatlabEngine();
    
    // establish a connection to MATLAB
    bool Connect(void);
    // check if a connection to MATLAB is established
    bool IsConnected(void) const;
    // close connection to MATLAB
    void CloseConnection(void);
    // close MATLAB
    void CloseMatlab(void);
    // check if MATLAB window is visible
    bool IsVisible(void) const;
    // set visibility of MATLAB
    void SetVisible(const bool visible);
    
    // execute a command in MATLAB
    void Execute(const char* command) const;
    
    // read an integer from MATLAB
    int GetInteger(const char* variableName) const;
    // write an integer to MATLAB
    void SetInteger(const char* variableName, const int value) const;
        
    // read an integer matrix from MATLAB
    int* GetIntegerMatrix(const char* variableName, unsigned long &noRows, unsigned long &noColumns) const;
    // read a double matrix from MATLAB
    double* GetDoubleMatrix(const char* variableName, unsigned long &noRows, unsigned long &noColumns) const;
    // write a double matrix to MATLAB
    void SetDoubleMatrix(const char* variableName, const double values[], const unsigned long noColumns, const unsigned long noRows) const;
    void SetIntegerMatrix(const char* variableName, const int values[], const unsigned long noColumns, const unsigned long noRows) const;
    
    // read a string from MATLAB
    const std::string GetString(const char* variableName) const;
    // write a string to MATLAB
    void WriteString(const char* string, const char* variableName) const;
    
    // check if a variable in MATLAB exists
    bool IsVariable(const char* variableName) const;
private:
//    Engine* _matlabEngine;
    mutable char _matlabEngineOutput[257];
protected:
};
