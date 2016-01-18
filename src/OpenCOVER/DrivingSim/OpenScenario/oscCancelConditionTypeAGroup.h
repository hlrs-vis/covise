/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_CANCEL_CONDITION_TYPE_A_GROUP_H
#define OSC_CANCEL_CONDITION_TYPE_A_GROUP_H

#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectArrayVariable.h>

#include <oscCancelConditionTypeA.h>


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscCancelConditionTypeAGroup: public oscObjectBase
{
public:
    oscCancelConditionTypeAGroup()
    {
        OSC_OBJECT_ADD_MEMBER(cancelCondition, "oscCancelConditionTypeA");
    };

    oscCancelConditionTypeAMember cancelCondition;
};

typedef oscObjectArrayVariable<oscCancelConditionTypeAGroup *> oscCancelConditionTypeAGroupArrayMember;

}

#endif //OSC_CANCEL_CONDITION_TYPE_A_GROUP_H
