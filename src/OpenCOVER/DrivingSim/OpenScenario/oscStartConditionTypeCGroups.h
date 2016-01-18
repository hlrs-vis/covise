/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_START_CONDITION_TYPE_C_GROUPS_H
#define OSC_START_CONDITION_TYPE_C_GROUPS_H

#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectArrayVariable.h>

#include <oscStartConditionTypeCGroup.h>


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscStartConditionTypeCGroups: public oscObjectBase
{
public:
    oscStartConditionTypeCGroups()
    {
        OSC_OBJECT_ADD_MEMBER(startConditionGroup, "oscStartConditionTypeCGroup");
    };

    oscStartConditionTypeCGroupArrayMember startConditionGroup;
};

typedef oscObjectArrayVariable<oscStartConditionTypeCGroups *> oscStartConditionTypeCGroupsArrayMember;

}

#endif /* OSC_START_CONDITION_TYPE_C_GROUPS_H */
