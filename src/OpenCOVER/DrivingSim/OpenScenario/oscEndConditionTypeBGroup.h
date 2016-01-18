/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_END_CONDITION_TYPE_B_GROUP_H
#define OSC_END_CONDITION_TYPE_B_GROUP_H

#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectArrayVariable.h>

#include <oscEndConditionTypeB.h>


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscEndConditionTypeBGroup: public oscObjectBase
{
public:
    oscEndConditionTypeBGroup()
    {
        OSC_OBJECT_ADD_MEMBER(endCondition, "oscEndConditionTypeB");
    };

    oscEndConditionTypeBMember endCondition;
};

typedef oscObjectArrayVariable<oscEndConditionTypeBGroup *> oscEndConditionTypeBGroupArrayMember;

}

#endif //OSC_END_CONDITION_TYPE_B_GROUP_H
