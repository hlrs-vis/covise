/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_START_CONDITION_TYPE_A_GROUPS_H
#define OSC_START_CONDITION_TYPE_A_GROUPS_H

#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectArrayVariable.h>

#include <oscStartConditionTypeAGroup.h>


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscStartConditionTypeAGroups: public oscObjectBase
{
public:
    oscStartConditionTypeAGroups()
    {
        OSC_OBJECT_ADD_MEMBER(startConditionGroup, "oscStartConditionTypeAGroup");
    };

    oscStartConditionTypeAGroupArrayMember startConditionGroup;
};

typedef oscObjectArrayVariable<oscStartConditionTypeAGroups *> oscStartConditionTypeAGroupsArrayMember;

}

#endif /* OSC_START_CONDITION_TYPE_A_GROUPS_H */
