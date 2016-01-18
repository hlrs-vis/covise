/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_START_CONDITION_TYPE_C_GROUP_H
#define OSC_START_CONDITION_TYPE_C_GROUP_H

#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectArrayVariable.h>

#include <oscStartConditionTypeC.h>


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscStartConditionTypeCGroup: public oscObjectBase
{
public:
    oscStartConditionTypeCGroup()
    {
        OSC_OBJECT_ADD_MEMBER(startCondition, "oscStartConditionTypeC");
    };

    oscStartConditionTypeCMember startCondition;
};

typedef oscObjectArrayVariable<oscStartConditionTypeCGroup *> oscStartConditionTypeCGroupArrayMember;

}

#endif //OSC_START_CONDITION_TYPE_C_GROUP_H
