/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCACT_H
#define OSCACT_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscSequence.h"
#include "oscActConditions.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscAct : public oscObjectBase
{
public:
oscAct()
{
        OSC_ADD_MEMBER(name, 0);
        OSC_OBJECT_ADD_MEMBER(Sequence, "oscSequence", 0);
        OSC_OBJECT_ADD_MEMBER(Conditions, "oscActConditions", 0);
    };
        const char *getScope(){return "/OpenSCENARIO/Storyboard/Story";};
    oscString name;
    oscSequenceMember Sequence;
    oscActConditionsMember Conditions;

};

typedef oscObjectVariable<oscAct *> oscActMember;
typedef oscObjectVariableArray<oscAct *> oscActArrayMember;


}

#endif //OSCACT_H
