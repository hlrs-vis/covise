/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCSTARTCONDITIONGROUP_H
#define OSCSTARTCONDITIONGROUP_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscCondition.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscStartConditionGroup : public oscObjectBase
{
public:
oscStartConditionGroup()
{
        OSC_OBJECT_ADD_MEMBER(Condition, "oscCondition", 0);
    };
        const char *getScope(){return "/OSCManeuver/Event/Conditions/Start";};
    oscConditionArrayMember Condition;

};

typedef oscObjectVariable<oscStartConditionGroup *> oscStartConditionGroupMember;
typedef oscObjectVariableArray<oscStartConditionGroup *> oscStartConditionGroupArrayMember;


}

#endif //OSCSTARTCONDITIONGROUP_H
