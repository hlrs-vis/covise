/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCAFTERTERMINATION_H
#define OSCAFTERTERMINATION_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscAtStart.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT Enum_AfterTermination_ruleType : public oscEnumType
{
public:
static Enum_AfterTermination_ruleType *instance();
    private:
		Enum_AfterTermination_ruleType();
	    static Enum_AfterTermination_ruleType *inst; 
};
class OPENSCENARIOEXPORT oscAfterTermination : public oscObjectBase
{
public:
oscAfterTermination()
{
        OSC_ADD_MEMBER(type, 0);
        OSC_ADD_MEMBER(name, 0);
        OSC_ADD_MEMBER(rule, 0);
        type.enumType = Enum_Story_Element_typeType::instance();
        rule.enumType = Enum_AfterTermination_ruleType::instance();
    };
    oscEnum type;
    oscString name;
    oscEnum rule;

    enum Enum_AfterTermination_rule
    {
end,
cancel,
any,

    };

};

typedef oscObjectVariable<oscAfterTermination *> oscAfterTerminationMember;
typedef oscObjectVariableArray<oscAfterTermination *> oscAfterTerminationArrayMember;


}

#endif //OSCAFTERTERMINATION_H
