/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCSTARTCONDITIONS_H
#define OSCSTARTCONDITIONS_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscConditionGroup.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscStartConditions : public oscObjectBase
{
public:
oscStartConditions()
{
        OSC_OBJECT_ADD_MEMBER(ConditionGroup, "oscConditionGroup", 0);
    };
        const char *getScope(){return "/OSCManeuver/Event";};
    oscConditionGroupArrayMember ConditionGroup;

};

typedef oscObjectVariable<oscStartConditions *> oscStartConditionsMember;
typedef oscObjectVariableArray<oscStartConditions *> oscStartConditionsArrayMember;


}

#endif //OSCSTARTCONDITIONS_H
