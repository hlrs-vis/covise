/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// time steps in COMSOL Multiphysics standard implementation
// author: André Buchau
// 19.10.2010: adaption to version 4.0a

#pragma once

#include "../include/time.hxx"
#include <vector>

class TimeStepsStandard : public TimeSteps
{
public:
    TimeStepsStandard(void);
    ~TimeStepsStandard();
    unsigned int getNoTimeSteps(void) const;
    double getValue(const unsigned int noTimeStep) const;
    Type getType() const;
    void setType(Type type);
    void putValue(const double t);
    void setFirstTimeStepOnly(const bool value);

private:
    std::vector<double> _timeSteps;
    Type _type;
    bool _firstTimeStepOnly;

protected:
};
