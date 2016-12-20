/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCPOSITION_H
#define OSCPOSITION_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscWorld.h"
#include "oscRelativeWorld.h"
#include "oscRelativeObject.h"
#include "oscRoad.h"
#include "oscRelativeRoad.h"
#include "oscLane.h"
#include "oscRelativeLane.h"
#include "oscPositionRoute.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscPosition : public oscObjectBase
{
public:
oscPosition()
{
        OSC_OBJECT_ADD_MEMBER(World, "oscWorld");
        OSC_OBJECT_ADD_MEMBER(RelativeWorld, "oscRelativeWorld");
        OSC_OBJECT_ADD_MEMBER(RelativeObject, "oscRelativeObject");
        OSC_OBJECT_ADD_MEMBER(Road, "oscRoad");
        OSC_OBJECT_ADD_MEMBER(RelativeRoad, "oscRelativeRoad");
        OSC_OBJECT_ADD_MEMBER(Lane, "oscLane");
        OSC_OBJECT_ADD_MEMBER(RelativeLane, "oscRelativeLane");
        OSC_OBJECT_ADD_MEMBER(PositionRoute, "oscPositionRoute");
    };
    oscWorldMember World;
    oscRelativeWorldMember RelativeWorld;
    oscRelativeObjectMember RelativeObject;
    oscRoadMember Road;
    oscRelativeRoadMember RelativeRoad;
    oscLaneMember Lane;
    oscRelativeLaneMember RelativeLane;
    oscPositionRouteMember PositionRoute;

};

typedef oscObjectVariable<oscPosition *> oscPositionMember;
typedef oscObjectVariableArray<oscPosition *> oscPositionArrayMember;


}

#endif //OSCPOSITION_H
