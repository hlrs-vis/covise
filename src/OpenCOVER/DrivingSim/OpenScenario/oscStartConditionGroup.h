/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_START_CONDITION_GROUP_H
#define OSC_START_CONDITION_GROUP_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscStartCondition.h>

namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscStartConditionGroup: public oscObjectBase
{
public:
    oscStartConditionGroup()
    {
		OSC_OBJECT_ADD_MEMBER(startCondition, "oscStartCondition");
    };
	oscStartConditionMember startCondition;
};

typedef oscObjectVariable<oscStartConditionGroup *> oscStartConditionGroupMember;

}

#endif //OSC_START_CONDITION_GROUP_H
