/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_POSITION_ROUTE_H
#define OSC_POSITION_ROUTE_H

#include "oscExport.h"
#include "oscOrientation.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "oscCurrentPosition.h"
#include "oscRoadCoord.h"
#include "oscLaneCoord.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscPositionRoute: public oscOrientation
{
public:
    oscPositionRoute()
    {
        OSC_ADD_MEMBER(routeId);
        OSC_ADD_MEMBER(relativeOrientation);
        OSC_OBJECT_ADD_MEMBER_CHOICE(currentPosition, "oscCurrentPosition");
        OSC_OBJECT_ADD_MEMBER_CHOICE(roadCoord, "oscRoadCoord");
        OSC_OBJECT_ADD_MEMBER_CHOICE(laneCoord, "oscLaneCoord");
    };

    oscString routeId;
    oscBool relativeOrientation;
    oscCurrentPositionMember currentPosition;
    oscRoadCoordMember roadCoord;
    oscLaneCoordMember laneCoord;

};

typedef oscObjectVariable<oscPositionRoute *> oscPositionRouteMember;

}

#endif //OSC_POSITION_ROUTE_H
