/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_START_CONDITION_TYPE_A_GROUP_H
#define OSC_START_CONDITION_TYPE_A_GROUP_H

#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectArrayVariable.h>

#include <oscStartConditionTypeA.h>


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscStartConditionTypeAGroup: public oscObjectBase
{
public:
    oscStartConditionTypeAGroup()
    {
        OSC_OBJECT_ADD_MEMBER(startCondition, "oscStartConditionTypeA");
    };

    oscStartConditionTypeAMember startCondition;
};

typedef oscObjectArrayVariable<oscStartConditionTypeAGroup *> oscStartConditionTypeAGroupArrayMember;

}

#endif //OSC_START_CONDITION_TYPE_A_GROUP_H
