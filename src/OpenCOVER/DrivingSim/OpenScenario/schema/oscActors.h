/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCACTORS_H
#define OSCACTORS_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscNamedEntity.h"
#include "oscByCondition.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscActors : public oscObjectBase
{
public:
oscActors()
{
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(NamedEntity, "oscNamedEntity");
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(ByCondition, "oscByCondition");
    };
    oscNamedEntityArrayMember NamedEntity;
    oscByConditionMember ByCondition;

};

typedef oscObjectVariable<oscActors *> oscActorsMember;
typedef oscObjectVariableArray<oscActors *> oscActorsArrayMember;


}

#endif //OSCACTORS_H
