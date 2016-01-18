/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_END_CONDITION_TYPE_A_GROUP_H
#define OSC_END_CONDITION_TYPE_A_GROUP_H

#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectArrayVariable.h>

#include <oscEndConditionTypeA.h>


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscEndConditionTypeAGroup: public oscObjectBase
{
public:
    oscEndConditionTypeAGroup()
    {
        OSC_OBJECT_ADD_MEMBER(endCondition, "oscEndConditionTypeA");
    };

    oscEndConditionTypeAMember endCondition;
};

typedef oscObjectArrayVariable<oscEndConditionTypeAGroup *> oscEndConditionTypeAGroupArrayMember;

}

#endif //OSC_END_CONDITION_TYPE_A_GROUP_H
