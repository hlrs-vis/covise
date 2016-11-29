/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCPOSITION_H
#define OSCPOSITION_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscWorld.h"
#include "schema/oscRelativeWorld.h"
#include "schema/oscRelativeObject.h"
#include "schema/oscRoad.h"
#include "schema/oscRelativeRoad.h"
#include "schema/oscLane.h"
#include "schema/oscRelativeLane.h"
#include "schema/oscPositionRoute.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscPosition : public oscObjectBase
{
public:
    oscPosition()
    {
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(World, "oscWorld");
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(RelativeWorld, "oscRelativeWorld");
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(RelativeObject, "oscRelativeObject");
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Road, "oscRoad");
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(RelativeRoad, "oscRelativeRoad");
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Lane, "oscLane");
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(RelativeLane, "oscRelativeLane");
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(PositionRoute, "oscPositionRoute");
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


}

#endif //OSCPOSITION_H
