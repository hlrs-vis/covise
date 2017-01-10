/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCPHASE_H
#define OSCPHASE_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscSignal.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscPhase : public oscObjectBase
{
public:
oscPhase()
{
        OSC_ADD_MEMBER(type);
        OSC_ADD_MEMBER(duration);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Signal, "oscSignal");
    };
    oscString type;
    oscDouble duration;
    oscSignalArrayMember Signal;

};

typedef oscObjectVariable<oscPhase *> oscPhaseMember;
typedef oscObjectVariableArray<oscPhase *> oscPhaseArrayMember;


}

#endif //OSCPHASE_H
