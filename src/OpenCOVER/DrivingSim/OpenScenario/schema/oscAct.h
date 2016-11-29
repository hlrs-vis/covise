/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCACT_H
#define OSCACT_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscSequence.h"
#include "schema/oscConditions.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscAct : public oscObjectBase
{
public:
    oscAct()
    {
        OSC_ADD_MEMBER(name);
        OSC_OBJECT_ADD_MEMBER(Sequence, "oscSequence");
        OSC_OBJECT_ADD_MEMBER(Conditions, "oscConditions");
    };
    oscString name;
    oscSequenceMember Sequence;
    oscConditionsMember Conditions;

};

typedef oscObjectVariable<oscAct *> oscActMember;


}

#endif //OSCACT_H
