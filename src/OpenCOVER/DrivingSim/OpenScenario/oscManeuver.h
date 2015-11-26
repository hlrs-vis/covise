/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_MANEUVER_H
#define OSC_MANEUVER_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>
#include <oscHeader.h>

namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscManeuver: public oscObjectBase
{
public:
    oscManeuver()
    {
        OSC_OBJECT_ADD_MEMBER(header,"oscHeader");
		OSC_ADD_MEMBER(name);
    };
    oscHeaderMember header;
	oscString name;
};

typedef oscObjectVariable<oscManeuver *>oscManeuverMember;

}

#endif //OSC_MANEUVER_H
