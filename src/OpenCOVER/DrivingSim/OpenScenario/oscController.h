/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_CONTROLLER_H
#define OSC_CONTROLLER_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "oscReference.h"
#include "oscPhase.h"

namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscController: public oscObjectBase
{
public:
    oscController()
    {
		OSC_ADD_MEMBER(name);
		OSC_ADD_MEMBER(delay);
		OSC_OBJECT_ADD_MEMBER(reference, "oscReference");
		OSC_OBJECT_ADD_MEMBER(Phase, "oscPhase");
    };

    oscString name;
	oscDouble delay;
	oscReferenceMember reference;
	oscPhaseMember Phase;
};

typedef oscObjectVariable<oscController *> oscControllerMember;

}

#endif //OSC_CONTROLLER_H
