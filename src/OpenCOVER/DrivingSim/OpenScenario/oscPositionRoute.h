/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_POSITION_ROUTE_H
#define OSC_POSITION_ROUTE_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>
#include <oscRoadCoord.h>
#include <oscLaneCoord.h>
#include <oscOrientation.h>

namespace OpenScenario {

class OpenScenarioBase;

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscPositionRoute: public oscObjectBase
{
public:
    oscPositionRoute()
    {
        OSC_ADD_MEMBER(routeId);
		OSC_ADD_MEMBER(current);
		OSC_OBJECT_ADD_MEMBER(roadCoord,"oscRoadCoord");
		OSC_OBJECT_ADD_MEMBER(laneCoord,"oscLaneCoord");
		OSC_ADD_MEMBER(relativeOrientation);
		OSC_OBJECT_ADD_MEMBER(orientation,"oscOrientation");
    };
    oscString routeId;
	oscBool current;
	oscRoadCoordMember roadCoord;
	oscLaneCoordMember laneCoord;
	oscBool relativeOrientation;
	oscOrientationMember orientation;
};

typedef oscObjectVariable<oscPositionRoute *> oscPositionRouteMember;

}

#endif //OSC_POSITION_ROUTE_H
