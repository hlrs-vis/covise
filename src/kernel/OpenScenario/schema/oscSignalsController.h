/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCSIGNALSCONTROLLER_H
#define OSCSIGNALSCONTROLLER_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscPhase.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscSignalsController : public oscObjectBase
{
public:
oscSignalsController()
{
        OSC_ADD_MEMBER(name, 0);
        OSC_ADD_MEMBER_OPTIONAL(delay, 0);
        OSC_ADD_MEMBER_OPTIONAL(reference, 0);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Phase, "oscPhase", 0);
    };
        const char *getScope(){return "/OpenSCENARIO/RoadNetwork/Signals";};
    oscString name;
    oscDouble delay;
    oscString reference;
    oscPhaseArrayMember Phase;

};

typedef oscObjectVariable<oscSignalsController *> oscSignalsControllerMember;
typedef oscObjectVariableArray<oscSignalsController *> oscSignalsControllerArrayMember;


}

#endif //OSCSIGNALSCONTROLLER_H
