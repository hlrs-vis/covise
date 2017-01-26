/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCMODIFY_H
#define OSCMODIFY_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscRule.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscModify : public oscObjectBase
{
public:
oscModify()
{
        OSC_OBJECT_ADD_MEMBER(Rule, "oscRule", 0);
    };
    oscRuleMember Rule;

};

typedef oscObjectVariable<oscModify *> oscModifyMember;
typedef oscObjectVariableArray<oscModify *> oscModifyArrayMember;


}

#endif //OSCMODIFY_H
