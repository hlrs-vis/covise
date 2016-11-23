/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_SIGNAL_H
#define OSC_SIGNAL_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscSignal: public oscObjectBase
{
public:
    oscSignal()
    {
		OSC_ADD_MEMBER(name);
		OSC_ADD_MEMBER(state);
    };

    oscString name;
	oscBool state;
};

typedef oscObjectVariable<oscSignal *> oscSignalMember;

}

#endif //OSC_SIGNAL_H
