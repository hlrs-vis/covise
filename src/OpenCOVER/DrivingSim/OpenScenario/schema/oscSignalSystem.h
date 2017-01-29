/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCSIGNALSYSTEM_H
#define OSCSIGNALSYSTEM_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscSetController.h"
#include "oscSetState.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscSignalSystem : public oscObjectBase
{
public:
oscSignalSystem()
{
        OSC_OBJECT_ADD_MEMBER(SetController, "oscSetController", 1);
        OSC_OBJECT_ADD_MEMBER(SetState, "oscSetState", 1);
    };
    oscSetControllerMember SetController;
    oscSetStateMember SetState;

};

typedef oscObjectVariable<oscSignalSystem *> oscSignalSystemMember;
typedef oscObjectVariableArray<oscSignalSystem *> oscSignalSystemArrayMember;


}

#endif //OSCSIGNALSYSTEM_H
