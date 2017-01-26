/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCCONDITIONS_H
#define OSCCONDITIONS_H

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
class OPENSCENARIOEXPORT oscConditions : public oscObjectBase
{
public:
oscConditions()
{
        OSC_OBJECT_ADD_MEMBER(Start, "oscStart", 0);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(End, "oscEnd", 0);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Cancel, "oscCancel", 0);
    };
    oscStartMember Start;
    oscEndMember End;
    oscCancelMember Cancel;

};

typedef oscObjectVariable<oscConditions *> oscConditionsMember;
typedef oscObjectVariableArray<oscConditions *> oscConditionsArrayMember;


}

#endif //OSCCONDITIONS_H
