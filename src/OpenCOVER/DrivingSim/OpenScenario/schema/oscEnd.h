/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCEND_H
#define OSCEND_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscConditionGroup.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscEnd : public oscObjectBase
{
public:
oscEnd()
{
        OSC_OBJECT_ADD_MEMBER(ConditionGroup, "oscConditionGroup", 0);
    };
    oscConditionGroupArrayMember ConditionGroup;

};

typedef oscObjectVariable<oscEnd *> oscEndMember;
typedef oscObjectVariableArray<oscEnd *> oscEndArrayMember;


}

#endif //OSCEND_H
