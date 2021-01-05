/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCACTIONCONTROLLER_H
#define OSCACTIONCONTROLLER_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscAssign.h"
#include "oscOverride.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscActionController : public oscObjectBase
{
public:
oscActionController()
{
        OSC_OBJECT_ADD_MEMBER(Assign, "oscAssign", 0);
        OSC_OBJECT_ADD_MEMBER(Override, "oscOverride", 0);
    };
        const char *getScope(){return "/OSCPrivateAction";};
    oscAssignMember Assign;
    oscOverrideMember Override;

};

typedef oscObjectVariable<oscActionController *> oscActionControllerMember;
typedef oscObjectVariableArray<oscActionController *> oscActionControllerArrayMember;


}

#endif //OSCACTIONCONTROLLER_H
