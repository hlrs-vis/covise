/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCSIGNALSYSTEM_H
#define OSCSIGNALSYSTEM_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscSetController.h"
#include "schema/oscSetState.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscSignalSystem : public oscObjectBase
{
public:
    oscSignalSystem()
    {
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(SetController, "oscSetController");
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(SetState, "oscSetState");
    };
    oscSetControllerMember SetController;
    oscSetStateMember SetState;

};

typedef oscObjectVariable<oscSignalSystem *> oscSignalSystemMember;


}

#endif //OSCSIGNALSYSTEM_H
