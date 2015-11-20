/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_SPEED_H
#define OSC_SPEED_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>
#include <oscSpeedDynamics.h>
#include <oscRelativeChoice.h>
#include <oscAbsolute.h>

namespace OpenScenario {

class OpenScenarioBase;

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscSpeed: public oscObjectBase
{
public:
    oscSpeed()
    {
		OSC_OBJECT_ADD_MEMBER(dynamics,"oscSpeedDynamics");
		OSC_OBJECT_ADD_MEMBER(relative,"oscRelativeChoice");
		OSC_OBJECT_ADD_MEMBER(absolute,"oscAbsolute");
    };
    oscSpeedDynamicsMember dynamics;
	oscRelativeChoiceMember relative;
	oscAbsoluteMember absolute;

};

typedef oscObjectVariable<oscSpeed *> oscSpeedMember;

}

#endif //OSC_SPEED_H
