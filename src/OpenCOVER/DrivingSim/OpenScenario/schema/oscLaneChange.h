/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCLANECHANGE_H
#define OSCLANECHANGE_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscChangeDynamics.h"
#include "schema/oscTarget.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscLaneChange : public oscObjectBase
{
public:
    oscLaneChange()
    {
        OSC_ADD_MEMBER(targetLaneOffset);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(ChangeDynamics, "oscChangeDynamics");
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Target, "oscTarget");
    };
    oscDouble targetLaneOffset;
    oscChangeDynamicsMember ChangeDynamics;
    oscTargetMember Target;

};

typedef oscObjectVariable<oscLaneChange *> oscLaneChangeMember;


}

#endif //OSCLANECHANGE_H
