/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCRELATIVESPEED_H
#define OSCRELATIVESPEED_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscTimeHeadway.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscRelativeSpeed : public oscObjectBase
{
public:
oscRelativeSpeed()
{
        OSC_ADD_MEMBER(entity);
        OSC_ADD_MEMBER(value);
        OSC_ADD_MEMBER(rule);
        rule.enumType = Enum_ruleType::instance();
    };
    oscString entity;
    oscDouble value;
    oscEnum rule;

};

typedef oscObjectVariable<oscRelativeSpeed *> oscRelativeSpeedMember;
typedef oscObjectVariableArray<oscRelativeSpeed *> oscRelativeSpeedArrayMember;


}

#endif //OSCRELATIVESPEED_H
