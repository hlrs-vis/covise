/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_POSITION_LANE_H
#define OSC_POSITION_LANE_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>
#include <oscOrientation.h>

namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscPositionLane: public oscObjectBase
{
public:
    oscPositionLane()
    {
        OSC_ADD_MEMBER(roadId);
		OSC_ADD_MEMBER(laneId);
		OSC_ADD_MEMBER(offset);
		OSC_ADD_MEMBER(s);
		OSC_ADD_MEMBER(relativeOrientation);
		OSC_OBJECT_ADD_MEMBER(orientation,"oscOrientation");
    };
    oscString roadId;
	oscInt laneId;
	oscDouble offset;
	oscDouble s;
	oscBool relativeOrientation;
	oscOrientationMember orientation;
};

typedef oscObjectVariable<oscPositionLane *> oscPositionLaneMember;

}

#endif //OSC_POSITION_LANE_H
