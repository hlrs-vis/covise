/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCSIGNALCONTROLLER_H
#define OSCSIGNALCONTROLLER_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscPhase.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscSignalController : public oscObjectBase
{
public:
oscSignalController()
{
        OSC_ADD_MEMBER(name, 0);
        OSC_ADD_MEMBER(delay, 0);
        OSC_ADD_MEMBER(reference, 0);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Phase, "oscPhase", 0);
    };
    oscString name;
    oscString delay;
    oscString reference;
    oscPhaseArrayMember Phase;

};

typedef oscObjectVariable<oscSignalController *> oscSignalControllerMember;
typedef oscObjectVariableArray<oscSignalController *> oscSignalControllerArrayMember;


}

#endif //OSCSIGNALCONTROLLER_H
