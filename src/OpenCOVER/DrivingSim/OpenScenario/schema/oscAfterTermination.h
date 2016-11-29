/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCAFTERTERMINATION_H
#define OSCAFTERTERMINATION_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscAtStart.h"

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
        OSC_ADD_MEMBER(type);
        OSC_ADD_MEMBER(name);
        OSC_ADD_MEMBER(rule);
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


}

#endif //OSCAFTERTERMINATION_H
