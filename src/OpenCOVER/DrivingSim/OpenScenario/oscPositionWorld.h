/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_POSITION_WORLD_H
#define OSC_POSITION_WORLD_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>
#include <oscCoordinate.h>
#include <oscOrientation.h>

namespace OpenScenario {

class OpenScenarioBase;

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscPisitionWorld: public oscObjectBase
{
public:
    oscPisitionWorld()
    {
        OSC_ADD_MEMBER(coordinate);
		OSC_ADD_MEMBER(orientation);		
    };
    oscCoordinateMember coordinate;
	oscOrientationMember orientation;
};

typedef oscObjectVariable<oscPisitionWorld *> oscPisitionWorldMember;

}

#endif //OSC_POSITION_WORLD_H
