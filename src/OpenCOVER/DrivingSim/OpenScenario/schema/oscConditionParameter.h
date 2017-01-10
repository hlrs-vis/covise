/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCCONDITIONPARAMETER_H
#define OSCCONDITIONPARAMETER_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscTimeHeadway.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscConditionParameter : public oscObjectBase
{
public:
oscConditionParameter()
{
        OSC_ADD_MEMBER(name);
        OSC_ADD_MEMBER(value);
        OSC_ADD_MEMBER(rule);
        rule.enumType = Enum_ruleType::instance();
    };
    oscString name;
    oscString value;
    oscEnum rule;

};

typedef oscObjectVariable<oscConditionParameter *> oscConditionParameterMember;
typedef oscObjectVariableArray<oscConditionParameter *> oscConditionParameterArrayMember;


}

#endif //OSCCONDITIONPARAMETER_H
