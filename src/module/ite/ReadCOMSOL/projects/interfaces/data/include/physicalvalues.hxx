// data container for physical values in the nodes of a problem
// author: A. Buchau
// modified: 18.11.2010

#pragma once
#include <vector>
#include "../sourcecode/dllapi.h"

class API_DATA PhysicalValues
{
public:
    PhysicalValues();
    virtual ~PhysicalValues();
    static PhysicalValues* getInstance(const unsigned int noTimeSteps, const std::vector <unsigned long> &noEvaluationPointsTimeStep);
    virtual void addDataSet(const std::string name, const bool vector) = 0;
    virtual unsigned int getNoEvaluationPoints(const unsigned int noTimeStep) const = 0;
    virtual unsigned int getNoDataSets() const = 0;
    virtual std::string getNameDataSet(const unsigned int no) const = 0;
    virtual bool isVectorDataSet(const unsigned int no) const = 0;
    virtual const double* getValue(const unsigned int noTimeStep, const unsigned int noDataSet, const unsigned int noValue) const = 0;
    virtual void setValue(const unsigned int noTimeStep, const unsigned int noDataSet, const unsigned int noValue, const double value[3]) = 0;
    virtual double* getAllValues(const unsigned int noTimeStep, const unsigned int noDataSet) = 0;
protected:
private:
};
