/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCLANE_H
#define OSCLANE_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"
#include "oscObjectVariableArray.h"

#include "oscVariables.h"
#include "schema/oscOrientation.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscLane : public oscObjectBase
{
public:
oscLane()
{
        OSC_ADD_MEMBER(roadId);
        OSC_ADD_MEMBER(laneId);
        OSC_ADD_MEMBER_OPTIONAL(offset);
        OSC_ADD_MEMBER(s);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Orientation, "oscOrientation");
    };
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
