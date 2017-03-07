/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCADDRULE_H
#define OSCADDRULE_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscAddRule : public oscObjectBase
{
public:
oscAddRule()
{
        OSC_ADD_MEMBER(value, 0);
    };
        const char *getScope(){return "/OSCGlobalAction/ActionParameter/Modify/Rule";};
    oscDouble value;

};

typedef oscObjectVariable<oscAddRule *> oscAddRuleMember;
typedef oscObjectVariableArray<oscAddRule *> oscAddRuleArrayMember;


}

#endif //OSCADDRULE_H
