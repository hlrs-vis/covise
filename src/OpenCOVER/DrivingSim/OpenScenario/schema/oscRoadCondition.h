/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCROADCONDITION_H
#define OSCROADCONDITION_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"
#include "oscObjectVariableArray.h"

#include "oscVariables.h"
#include "schema/oscEffect.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscRoadCondition : public oscObjectBase
{
public:
oscRoadCondition()
{
        OSC_ADD_MEMBER(frictionScale);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Effect, "oscEffect");
    };
    oscDouble frictionScale;
    oscEffectArrayMember Effect;

};

typedef oscObjectVariable<oscRoadCondition *> oscRoadConditionMember;
typedef oscObjectVariableArray<oscRoadCondition *> oscRoadConditionArrayMember;


}

#endif //OSCROADCONDITION_H
