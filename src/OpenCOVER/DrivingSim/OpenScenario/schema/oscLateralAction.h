/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCLATERALACTION_H
#define OSCLATERALACTION_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscLaneChange.h"
#include "schema/oscLaneOffset.h"
#include "schema/oscDistance.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscLateralAction : public oscObjectBase
{
public:
    oscLateralAction()
    {
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(LaneChange, "oscLaneChange");
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(LaneOffset, "oscLaneOffset");
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Distance, "oscDistance");
    };
    oscLaneChangeMember LaneChange;
    oscLaneOffsetMember LaneOffset;
    oscDistanceMember Distance;

};

typedef oscObjectVariable<oscLateralAction *> oscLateralActionMember;


}

#endif //OSCLATERALACTION_H
