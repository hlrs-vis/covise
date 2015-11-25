/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_POSITION_ROAD_H
#define OSC_POSITION_ROAD_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>
#include <oscOrientation.h>

namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscPositionRoad: public oscObjectBase
{
public:
    oscPositionRoad()
    {
        OSC_ADD_MEMBER(roadId);
		OSC_ADD_MEMBER(s);
		OSC_ADD_MEMBER(t);
		OSC_ADD_MEMBER(relativeOrientation);
		OSC_OBJECT_ADD_MEMBER(orientation,"oscOrientation");
    };
    oscString roadId;
	oscDouble s;
	oscDouble t;
	oscBool relativeOrientation;
	oscOrientationMember orientation;
};

typedef oscObjectVariable<oscPositionRoad *> oscPositionRoadMember;

}

#endif //OSC_POSITION_ROAD_H
