/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_ACTION_H
#define OSC_ACTION_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>
#include <oscNamedObject.h>
#include <oscAutonomous.h>
#include <oscSpeed.h>
#include <oscLaneChange.h>

namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscAction: public oscObjectBase
{
public:
    oscAction()
    {
        OSC_OBJECT_ADD_MEMBER(name,"oscNamedObject");
		OSC_OBJECT_ADD_MEMBER(autonomous,"oscAutonomous");
		OSC_OBJECT_ADD_MEMBER(speed,"oscSpeed");
		OSC_OBJECT_ADD_MEMBER(laneChange,"oscLaneChange");
    };
    oscNamedObjectMember name;
	oscAutonomousMember autonomous;
	oscSpeedMember speed;
	oscLaneChangeMember laneChange;
};

typedef oscObjectVariable<oscAction *> oscActionMember;

}

#endif //OSC_ACTION_H
