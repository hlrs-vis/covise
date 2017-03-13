/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCLANECHANGE_H
#define OSCLANECHANGE_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscLaneChangeDynamics.h"
#include "oscLaneChangeTarget.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscLaneChange : public oscObjectBase
{
public:
oscLaneChange()
{
        OSC_ADD_MEMBER_OPTIONAL(targetLaneOffset, 0);
        OSC_OBJECT_ADD_MEMBER(Dynamics, "oscLaneChangeDynamics", 0);
        OSC_OBJECT_ADD_MEMBER(Target, "oscLaneChangeTarget", 0);
    };
        const char *getScope(){return "/OSCPrivateAction/Lateral";};
    oscDouble targetLaneOffset;
    oscLaneChangeDynamicsMember Dynamics;
    oscLaneChangeTargetMember Target;

};

typedef oscObjectVariable<oscLaneChange *> oscLaneChangeMember;
typedef oscObjectVariableArray<oscLaneChange *> oscLaneChangeArrayMember;


}

#endif //OSCLANECHANGE_H
