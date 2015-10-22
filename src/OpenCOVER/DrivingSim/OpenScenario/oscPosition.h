/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_POSITION_H
#define OSC_POSITION_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>
#include <oscPositionWorld.h>
#include <oscPositionRoad.h>
#include <oscPositionLane.h>
#include <oscRelativePositionWorld.h>
#include <oscRelativePositionRoad.h>
#include <oscRelativePositionLane.h>
#include <oscPositionRoute.h>

namespace OpenScenario {

class OpenScenarioBase;

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscPosition: public oscObjectBase
{
public:
    oscPosition()
    {
        OSC_ADD_MEMBER(positionWorld);
		OSC_ADD_MEMBER(positionRoad);
		OSC_ADD_MEMBER(positionLane);
		OSC_ADD_MEMBER(relativePositionWorld);
		OSC_ADD_MEMBER(relativePositionRoad);
		OSC_ADD_MEMBER(relativePositionLane);
		OSC_ADD_MEMBER(positionRoute);
    };
    oscPositionWorldMember positionWorld;
	oscPositionRoadMember positionRoad;
	oscPositionLaneMember positionLane;
	oscRelativePisitionWorldMember relativePositionWorld;
	oscRelativePositionRoadMember relativePositionRoad;
	oscRelativePositionLaneMember relativePositionLane;
	oscPositionRouteMember positionRoute;
};

typedef oscObjectVariable<oscPosition *> oscPositionMember;

}

#endif //OSC_POSITION_H
