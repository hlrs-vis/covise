/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCLANE_H
#define OSCLANE_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscOrientation.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscLane : public oscObjectBase
{
public:
oscLane()
{
        OSC_ADD_MEMBER(roadId, 0);
        OSC_ADD_MEMBER(laneId, 0);
        OSC_ADD_MEMBER_OPTIONAL(offset, 0);
        OSC_ADD_MEMBER(s, 0);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Orientation, "oscOrientation", 0);
    };
        const char *getScope(){return "/OSCPosition";};
    oscString roadId;
    oscInt laneId;
    oscDouble offset;
    oscDouble s;
    oscOrientationMember Orientation;

};

typedef oscObjectVariable<oscLane *> oscLaneMember;
typedef oscObjectVariableArray<oscLane *> oscLaneArrayMember;


}

#endif //OSCLANE_H
