/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCLANEOFFSET_H
#define OSCLANEOFFSET_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscOffsetDynamics.h"
#include "schema/oscTarget.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscLaneOffset : public oscObjectBase
{
public:
    oscLaneOffset()
    {
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(OffsetDynamics, "oscOffsetDynamics");
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Target, "oscTarget");
    };
    oscOffsetDynamicsMember OffsetDynamics;
    oscTargetMember Target;

};

typedef oscObjectVariable<oscLaneOffset *> oscLaneOffsetMember;


}

#endif //OSCLANEOFFSET_H
