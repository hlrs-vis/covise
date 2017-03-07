/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCENVIRONMENTTIMEOFDAY_H
#define OSCENVIRONMENTTIMEOFDAY_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscTime.h"
#include "oscDate.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscEnvironmentTimeOfDay : public oscObjectBase
{
public:
oscEnvironmentTimeOfDay()
{
        OSC_ADD_MEMBER(animation, 0);
        OSC_OBJECT_ADD_MEMBER(Time, "oscTime", 0);
        OSC_OBJECT_ADD_MEMBER(Date, "oscDate", 0);
    };
        const char *getScope(){return "/OSCEnvironment";};
    oscBool animation;
    oscTimeMember Time;
    oscDateMember Date;

};

typedef oscObjectVariable<oscEnvironmentTimeOfDay *> oscEnvironmentTimeOfDayMember;
typedef oscObjectVariableArray<oscEnvironmentTimeOfDay *> oscEnvironmentTimeOfDayArrayMember;


}

#endif //OSCENVIRONMENTTIMEOFDAY_H
