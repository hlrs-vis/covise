/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_END_CONDITION_TYPE_A_GROUPS_H
#define OSC_END_CONDITION_TYPE_A_GROUPS_H

#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectArrayVariable.h>

#include <oscEndConditionTypeAGroup.h>


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscEndConditionTypeAGroups: public oscObjectBase
{
public:
    oscEndConditionTypeAGroups()
    {
        OSC_OBJECT_ADD_MEMBER(endConditionGroup, "oscEndConditionGroup");
    };

    oscEndConditionTypeAGroupArrayMember endConditionGroup;
};

typedef oscObjectArrayVariable<oscEndConditionTypeAGroups *> oscEndConditionTypeAGroupsArrayMember;

}

#endif /* OSC_END_CONDITION_TYPE_A_GROUPS_H */
