/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_POSITION_H
#define OSC_POSITION_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscPositionWorld.h>
#include <oscPositionRoad.h>
#include <oscPositionLane.h>
#include <oscRelativePositionWorld.h>
#include <oscRelativePositionRoad.h>
#include <oscRelativePositionLane.h>
#include <oscPositionRoute.h>

namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscPosition: public oscObjectBase
{
public:
    oscPosition()
    {
        OSC_OBJECT_ADD_MEMBER(positionWorld,"oscPositionWorld");
        OSC_OBJECT_ADD_MEMBER(positionRoad,"oscPositionRoad");
		OSC_OBJECT_ADD_MEMBER(positionLane,"oscPositionLane");
		OSC_OBJECT_ADD_MEMBER(relativePositionWorld,"oscRelativePositionWorld");
		OSC_OBJECT_ADD_MEMBER(relativePositionRoad,"oscRelativePositionRoad");
		OSC_OBJECT_ADD_MEMBER(relativePositionLane,"oscRelativePositionLane");
		OSC_OBJECT_ADD_MEMBER(positionRoute,"oscPositionRoute");
    };
    oscPositionWorldMember positionWorld;
	oscPositionRoadMember positionRoad;
	oscPositionLaneMember positionLane;
	oscRelativePositionWorldMember relativePositionWorld;
	oscRelativePositionRoadMember relativePositionRoad;
	oscRelativePositionLaneMember relativePositionLane;
	oscPositionRouteMember positionRoute;
};

typedef oscObjectVariable<oscPosition *> oscPositionMember;

}

#endif //OSC_POSITION_H
