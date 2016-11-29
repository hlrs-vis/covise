/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCCONDITIONGROUP_H
#define OSCCONDITIONGROUP_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscCondition.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscConditionGroup : public oscObjectBase
{
public:
    oscConditionGroup()
    {
        OSC_OBJECT_ADD_MEMBER(Condition, "oscCondition");
    };
    oscConditionMember Condition;

};

typedef oscObjectVariable<oscConditionGroup *> oscConditionGroupMember;


}

#endif //OSCCONDITIONGROUP_H
