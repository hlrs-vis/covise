/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __DTF_TIMESTEPS_H_
#define __DTF_TIMESTEPS_H_

#include "timestep.h"

namespace DTF
{
class ClassInfo_DTFTimeSteps;

class TimeSteps : public Tools::BaseObject
{
    friend class ClassInfo_DTFTimeSteps;

private:
    TimeSteps();
    TimeSteps(string className, int objectID);

    bool fillTimeStep(TimeStep *timeStep, Data *data);

    static ClassInfo_DTFTimeSteps classInfo;

    vector<TimeStep *> timeSteps;

public:
    virtual ~TimeSteps();

    TimeStep *get(int timeStepNum);
    int getNumTimeSteps();
    bool setData(vector<Data *> data);

    virtual void clear();
    virtual void print();
};

CLASSINFO(ClassInfo_DTFTimeSteps, TimeSteps);
};
#endif
