/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCRELATIVELANE_H
#define OSCRELATIVELANE_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscOrientation.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscRelativeLane : public oscObjectBase
{
public:
oscRelativeLane()
{
        OSC_ADD_MEMBER(object, 0);
        OSC_ADD_MEMBER(dLane, 0);
        OSC_ADD_MEMBER(ds, 0);
        OSC_ADD_MEMBER_OPTIONAL(offset, 0);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Orientation, "oscOrientation", 0);
    };
    oscString object;
    oscInt dLane;
    oscDouble ds;
    oscDouble offset;
    oscOrientationMember Orientation;

};

typedef oscObjectVariable<oscRelativeLane *> oscRelativeLaneMember;
typedef oscObjectVariableArray<oscRelativeLane *> oscRelativeLaneArrayMember;


}

#endif //OSCRELATIVELANE_H
