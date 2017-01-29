/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCROUTEPOSITION_H
#define OSCROUTEPOSITION_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscCurrent.h"
#include "oscRoadCoord.h"
#include "oscLaneCoord.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscRoutePosition : public oscObjectBase
{
public:
oscRoutePosition()
{
        OSC_OBJECT_ADD_MEMBER(Current, "oscCurrent", 1);
        OSC_OBJECT_ADD_MEMBER(RoadCoord, "oscRoadCoord", 1);
        OSC_OBJECT_ADD_MEMBER(LaneCoord, "oscLaneCoord", 1);
    };
    oscCurrentMember Current;
    oscRoadCoordMember RoadCoord;
    oscLaneCoordMember LaneCoord;

};

typedef oscObjectVariable<oscRoutePosition *> oscRoutePositionMember;
typedef oscObjectVariableArray<oscRoutePosition *> oscRoutePositionArrayMember;


}

#endif //OSCROUTEPOSITION_H
