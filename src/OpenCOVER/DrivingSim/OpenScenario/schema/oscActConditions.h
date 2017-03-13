/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCACTCONDITIONS_H
#define OSCACTCONDITIONS_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscStart.h"
#include "oscEnd.h"
#include "oscCancel.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscActConditions : public oscObjectBase
{
public:
oscActConditions()
{
        OSC_OBJECT_ADD_MEMBER(Start, "oscStart", 0);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(End, "oscEnd", 0);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Cancel, "oscCancel", 0);
    };
        const char *getScope(){return "/OpenSCENARIO/Storyboard/Story/Act";};
    oscStartMember Start;
    oscEndMember End;
    oscCancelMember Cancel;

};

typedef oscObjectVariable<oscActConditions *> oscActConditionsMember;
typedef oscObjectVariableArray<oscActConditions *> oscActConditionsArrayMember;


}

#endif //OSCACTCONDITIONS_H
