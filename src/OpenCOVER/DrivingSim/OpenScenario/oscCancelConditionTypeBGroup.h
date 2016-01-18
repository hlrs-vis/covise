/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_CANCEL_CONDITION_TYPE_B_GROUP_H
#define OSC_CANCEL_CONDITION_TYPE_B_GROUP_H

#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectArrayVariable.h>

#include <oscCancelConditionTypeB.h>


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscCancelConditionTypeBGroup: public oscObjectBase
{
public:
    oscCancelConditionTypeBGroup()
    {
        OSC_OBJECT_ADD_MEMBER(cancelCondition, "oscCancelConditionTypeB");
    };

    oscCancelConditionTypeBMember cancelCondition;
};

typedef oscObjectArrayVariable<oscCancelConditionTypeBGroup *> oscCancelConditionTypeBGroupArrayMember;

}

#endif //OSC_CANCEL_CONDITION_TYPE_B_GROUP_H
