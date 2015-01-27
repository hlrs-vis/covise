/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "timestandard.h"

TimeStepsStandard::TimeStepsStandard(void)
    : _firstTimeStepOnly(false)
    , _type(Type_static)
{
    _timeSteps.resize(0);
}

TimeStepsStandard::~TimeStepsStandard()
{
}

unsigned int TimeStepsStandard::getNoTimeSteps(void) const
{
    unsigned int retVal = 0;
    if (_firstTimeStepOnly)
        retVal = 1;
    else
        retVal = _timeSteps.size();
    return retVal;
}

double TimeStepsStandard::getValue(const unsigned int noTimeStep) const
{
    double retVal = 0;
    if (noTimeStep < _timeSteps.size())
        retVal = _timeSteps[noTimeStep];
    return retVal;
}

TimeStepsStandard::Type TimeStepsStandard::getType() const
{
    return _type;
}

void TimeStepsStandard::setType(Type type)
{
    _type = type;
}

void TimeStepsStandard::putValue(const double t)
{
    _timeSteps.push_back(t);
}

void TimeStepsStandard::setFirstTimeStepOnly(const bool value)
{
    _firstTimeStepOnly = value;
}
