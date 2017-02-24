/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCLATERAL_H
#define OSCLATERAL_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscLaneChange.h"
#include "oscLaneOffset.h"
#include "oscDistance.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscLateral : public oscObjectBase
{
public:
oscLateral()
{
        OSC_OBJECT_ADD_MEMBER(LaneChange, "oscLaneChange", 1);
        OSC_OBJECT_ADD_MEMBER(LaneOffset, "oscLaneOffset", 1);
        OSC_OBJECT_ADD_MEMBER(Distance, "oscDistance", 1);
    };
        const char *getScope(){return "/OSCPrivateAction";};
    oscLaneChangeMember LaneChange;
    oscLaneOffsetMember LaneOffset;
    oscDistanceMember Distance;

};

typedef oscObjectVariable<oscLateral *> oscLateralMember;
typedef oscObjectVariableArray<oscLateral *> oscLateralArrayMember;


}

#endif //OSCLATERAL_H
