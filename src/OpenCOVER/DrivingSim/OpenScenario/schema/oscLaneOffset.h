/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCLANEOFFSET_H
#define OSCLANEOFFSET_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscLaneOffsetDynamics.h"
#include "oscLaneOffsetTarget.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscLaneOffset : public oscObjectBase
{
public:
oscLaneOffset()
{
        OSC_OBJECT_ADD_MEMBER(Dynamics, "oscLaneOffsetDynamics", 0);
        OSC_OBJECT_ADD_MEMBER(Target, "oscLaneOffsetTarget", 0);
    };
        const char *getScope(){return "/OSCPrivateAction/Lateral";};
    oscLaneOffsetDynamicsMember Dynamics;
    oscLaneOffsetTargetMember Target;

};

typedef oscObjectVariable<oscLaneOffset *> oscLaneOffsetMember;
typedef oscObjectVariableArray<oscLaneOffset *> oscLaneOffsetArrayMember;


}

#endif //OSCLANEOFFSET_H
