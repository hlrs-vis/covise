/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "physicalvaluesstandard.h"

PhysicalValuesStandard::PhysicalValuesStandard(const unsigned int noTimeSteps, const std::vector<unsigned long> &noEvaluationPoints)
    : _noTimeSteps(noTimeSteps)
{
    _noEvaluationPoints.resize(_noTimeSteps);
    for (unsigned int i = 0; i < _noTimeSteps; i++)
    {
        if (i < noEvaluationPoints.size())
            _noEvaluationPoints[i] = noEvaluationPoints[i];
        else
            _noEvaluationPoints[i] = 0;
    }
    _dataSets.resize(0);
}

PhysicalValuesStandard::~PhysicalValuesStandard()
{
}

void PhysicalValuesStandard::addDataSet(const std::string name, const bool vector)
{
    const unsigned int index = _dataSets.size();
    _dataSets.resize(index + 1);
    _dataSets[index].name = name;
    _dataSets[index].vector = vector;
    _dataSets[index].values.resize(_noTimeSteps);
    for (unsigned int i = 0; i < _noTimeSteps; i++)
    {
        if (vector)
            _dataSets[index].values[i].resize(_noEvaluationPoints[i] * 3);
        else
            _dataSets[index].values[i].resize(_noEvaluationPoints[i]);
    }
}

unsigned int PhysicalValuesStandard::getNoEvaluationPoints(const unsigned int noTimeStep) const
{
    unsigned int retVal = 0;
    if (noTimeStep < _noTimeSteps)
        retVal = _noEvaluationPoints[noTimeStep];
    return retVal;
}

unsigned int PhysicalValuesStandard::getNoDataSets() const
{
    return _dataSets.size();
}

std::string PhysicalValuesStandard::getNameDataSet(const unsigned int no) const
{
    std::string retVal = "";
    if (no < _dataSets.size())
        retVal = _dataSets[no].name;
    return retVal;
}

bool PhysicalValuesStandard::isVectorDataSet(const unsigned int no) const
{
    bool retVal = false;
    if (no < _dataSets.size())
        retVal = _dataSets[no].vector;
    return retVal;
}

const double *PhysicalValuesStandard::getValue(const unsigned int noTimeStep, const unsigned int noDataSet, const unsigned int noValue) const
{
    double const *retVal = 0;
    if (noDataSet < _dataSets.size())
    {
        if (noTimeStep < _dataSets[noDataSet].values.size())
        {
            if (noValue < _dataSets[noDataSet].values[noTimeStep].size())
            {
                unsigned int index = 0;
                if (isVectorDataSet(noDataSet))
                    index = noValue * 3;
                else
                    index = noValue;
                retVal = &_dataSets[noDataSet].values[noTimeStep][index];
            }
        }
    }
    return retVal;
}

void PhysicalValuesStandard::setValue(const unsigned int noTimeStep, const unsigned int noDataSet, const unsigned int noValue, const double value[3])
{
    if (noDataSet < _dataSets.size())
    {
        if (noTimeStep < _dataSets[noDataSet].values.size())
        {
            if (noValue < _dataSets[noDataSet].values[noTimeStep].size())
            {
                if (isVectorDataSet(noDataSet))
                {
                    for (unsigned char i = 0; i < 3; i++)
                        _dataSets[noDataSet].values[noTimeStep][noValue * 3 + i] = value[i];
                }
                else
                    _dataSets[noDataSet].values[noTimeStep][noValue] = value[0];
            }
        }
    }
}

double *PhysicalValuesStandard::getAllValues(const unsigned int noTimeStep, const unsigned int noDataSet)
{
    double *retVal = 0;
    if (noDataSet < _dataSets.size())
    {
        if (noTimeStep < _dataSets[noDataSet].values.size())
            retVal = &_dataSets[noDataSet].values[noTimeStep][0];
    }
    return retVal;
}
