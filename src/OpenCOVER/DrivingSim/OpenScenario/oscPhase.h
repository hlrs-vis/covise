/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_PHASE_H
#define OSC_PHASE_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "oscSignal.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscPhase: public oscObjectBase
{
public:
    oscPhase()
    {
		OSC_ADD_MEMBER(type);
		OSC_ADD_MEMBER(duration);
		OSC_OBJECT_ADD_MEMBER(Signal, "oscSignal");
    };

    oscString type;
	oscDouble duration;
	oscSignalMember Signal;

};

typedef oscObjectVariable<oscPhase *> oscPhaseMember;

}

#endif //OSC_PHASE_H
