/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCPOSITIONROUTE_H
#define OSCPOSITIONROUTE_H

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
class OPENSCENARIOEXPORT oscPositionRoute : public oscObjectBase
{
public:
oscPositionRoute()
{
        OSC_OBJECT_ADD_MEMBER(Current, "oscCurrent", 1);
        OSC_OBJECT_ADD_MEMBER(RoadCoord, "oscRoadCoord", 1);
        OSC_OBJECT_ADD_MEMBER(LaneCoord, "oscLaneCoord", 1);
    };
        const char *getScope(){return "/OSCPosition/Route";};
    oscCurrentMember Current;
    oscRoadCoordMember RoadCoord;
    oscLaneCoordMember LaneCoord;

};

typedef oscObjectVariable<oscPositionRoute *> oscPositionRouteMember;
typedef oscObjectVariableArray<oscPositionRoute *> oscPositionRouteArrayMember;


}

#endif //OSCPOSITIONROUTE_H
