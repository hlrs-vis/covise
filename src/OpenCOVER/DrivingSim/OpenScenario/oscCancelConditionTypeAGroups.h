/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_CANCEL_CONDITION_TYPE_A_GROUPS_H
#define OSC_CANCEL_CONDITION_TYPE_A_GROUPS_H

#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectArrayVariable.h>

#include <oscCancelConditionTypeAGroup.h>


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscCancelConditionTypeAGroups: public oscObjectBase
{
public:
    oscCancelConditionTypeAGroups()
    {
        OSC_OBJECT_ADD_MEMBER(cancelConditionGroup, "oscCancelConditionTypeAGroup");
    };

    oscCancelConditionTypeAGroupArrayMember cancelConditionGroup;
};

typedef oscObjectArrayVariable<oscCancelConditionTypeAGroups *> oscCancelConditionTypeAGroupsArrayMember;

}

#endif /* OSC_CANCEL_CONDITION_TYPE_A_GROUPS_H */
