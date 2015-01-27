/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// data container for physical values in the nodes of a problem standard implementation
// author: A. Buchau
// modified: 18.11.2010

#pragma once
#include "../include/physicalvalues.hxx"

class PhysicalValuesStandard : public PhysicalValues
{
public:
    PhysicalValuesStandard(const unsigned int noTimeSteps, const std::vector<unsigned long> &noEvaluationPoints);
    ~PhysicalValuesStandard();
    void addDataSet(const std::string name, const bool vector);
    unsigned int getNoEvaluationPoints(const unsigned int noTimeStep) const;
    unsigned int getNoDataSets() const;
    std::string getNameDataSet(const unsigned int no) const;
    bool isVectorDataSet(const unsigned int no) const;
    const double *getValue(const unsigned int noTimeStep, const unsigned int noDataSet, const unsigned int noValue) const;
    void setValue(const unsigned int noTimeStep, const unsigned int noDataSet, const unsigned int noValue, const double value[3]);
    double *getAllValues(const unsigned int noTimeStep, const unsigned int noDataSet);

protected:
private:
    const unsigned int _noTimeSteps;
    std::vector<unsigned long> _noEvaluationPoints;
    struct DataSet
    {
        std::string name;
        bool vector;
        std::vector<std::vector<double> > values;
    };
    std::vector<DataSet> _dataSets;
};
