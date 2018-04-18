/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCENDCONDITIONS_H
#define OSCENDCONDITIONS_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscConditionGroup.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscEndConditions : public oscObjectBase
{
public:
oscEndConditions()
{
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(ConditionGroup, "oscConditionGroup", 0);
    };
        const char *getScope(){return "/OpenSCENARIO/Storyboard";};
    oscConditionGroupArrayMember ConditionGroup;

};

typedef oscObjectVariable<oscEndConditions *> oscEndConditionsMember;
typedef oscObjectVariableArray<oscEndConditions *> oscEndConditionsArrayMember;


}

#endif //OSCENDCONDITIONS_H
