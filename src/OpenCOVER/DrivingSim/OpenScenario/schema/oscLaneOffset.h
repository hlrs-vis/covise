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
#include "oscOffsetDynamics.h"
#include "oscTarget.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscLaneOffset : public oscObjectBase
{
public:
oscLaneOffset()
{
        OSC_OBJECT_ADD_MEMBER(OffsetDynamics, "oscOffsetDynamics");
        OSC_OBJECT_ADD_MEMBER(Target, "oscTarget");
    };
    oscOffsetDynamicsMember OffsetDynamics;
    oscTargetMember Target;

};

typedef oscObjectVariable<oscLaneOffset *> oscLaneOffsetMember;
typedef oscObjectVariableArray<oscLaneOffset *> oscLaneOffsetArrayMember;


}

#endif //OSCLANEOFFSET_H
