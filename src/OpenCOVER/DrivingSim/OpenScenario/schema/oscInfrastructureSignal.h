/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCINFRASTRUCTURESIGNAL_H
#define OSCINFRASTRUCTURESIGNAL_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscSetController.h"
#include "oscSetState.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscInfrastructureSignal : public oscObjectBase
{
public:
oscInfrastructureSignal()
{
        OSC_OBJECT_ADD_MEMBER(SetController, "oscSetController", 1);
        OSC_OBJECT_ADD_MEMBER(SetState, "oscSetState", 1);
    };
        const char *getScope(){return "/OSCGlobalAction/Infrastructure";};
    oscSetControllerMember SetController;
    oscSetStateMember SetState;

};

typedef oscObjectVariable<oscInfrastructureSignal *> oscInfrastructureSignalMember;
typedef oscObjectVariableArray<oscInfrastructureSignal *> oscInfrastructureSignalArrayMember;


}

#endif //OSCINFRASTRUCTURESIGNAL_H
