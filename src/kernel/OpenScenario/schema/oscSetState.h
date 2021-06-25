/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCSETSTATE_H
#define OSCSETSTATE_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscSetState : public oscObjectBase
{
public:
oscSetState()
{
        OSC_ADD_MEMBER(name, 0);
        OSC_ADD_MEMBER(state, 0);
    };
        const char *getScope(){return "/OSCGlobalAction/Infrastructure/InfrastructureSignal";};
    oscString name;
    oscString state;

};

typedef oscObjectVariable<oscSetState *> oscSetStateMember;
typedef oscObjectVariableArray<oscSetState *> oscSetStateArrayMember;


}

#endif //OSCSETSTATE_H
