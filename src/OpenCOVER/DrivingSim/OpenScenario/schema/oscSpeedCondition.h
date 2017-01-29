/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCSPEEDCONDITION_H
#define OSCSPEEDCONDITION_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscTimeHeadway.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscSpeedCondition : public oscObjectBase
{
public:
oscSpeedCondition()
{
        OSC_ADD_MEMBER(value, 0);
        OSC_ADD_MEMBER(rule, 0);
        rule.enumType = Enum_ruleType::instance();
    };
    oscDouble value;
    oscEnum rule;

};

typedef oscObjectVariable<oscSpeedCondition *> oscSpeedConditionMember;
typedef oscObjectVariableArray<oscSpeedCondition *> oscSpeedConditionArrayMember;


}

#endif //OSCSPEEDCONDITION_H
