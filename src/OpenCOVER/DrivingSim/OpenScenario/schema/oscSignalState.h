/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCSIGNALSTATE_H
#define OSCSIGNALSTATE_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscSignalState : public oscObjectBase
{
public:
oscSignalState()
{
        OSC_ADD_MEMBER(name);
        OSC_ADD_MEMBER(state);
    };
    oscString name;
    oscString state;

};

typedef oscObjectVariable<oscSignalState *> oscSignalStateMember;
typedef oscObjectVariableArray<oscSignalState *> oscSignalStateArrayMember;


}

#endif //OSCSIGNALSTATE_H
