// time steps in COMSOL Multiphysics
// author: André Buchau
// 19.10.2010: adaption to version 4.0a
// 18.11.2010: modified DLL

#pragma once

#include "../sourcecode/dllapi.h"

class API_DATA TimeSteps{
public:
    TimeSteps(void);
    virtual ~TimeSteps();
    static TimeSteps* getInstance();
    enum QualityTimeHarmonic
    {
        QualityTimeHarmonic_Low,
        QualityTimeHarmonic_Medium,
        QualityTimeHarmonic_High
    };
    enum Type
    {
        Type_static,
        Type_harmonic,
        Type_transient,
        Type_parametric
    };
    virtual unsigned int getNoTimeSteps(void) const = 0;
    virtual double getValue(const unsigned int noTimeStep) const = 0;
    virtual Type getType() const = 0;
    virtual void setType(Type type) = 0;
    virtual void putValue(const double t) = 0;
    virtual void setFirstTimeStepOnly(const bool value) = 0;
private:
protected:
};
