/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_END_CONDITIONS_TYPE_A_H
#define OSC_END_CONDITIONS_TYPE_A_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariableArray.h"

#include "oscConditionTypeA.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscEndConditionsTypeA: public oscObjectBase
{
public:
    oscEndConditionsTypeA()
    {
        OSC_OBJECT_ADD_MEMBER(endCondition, "oscConditionTypeA");
    };

    oscConditionTypeAMember endCondition;
};

typedef oscObjectVariableArray<oscEndConditionsTypeA *> oscEndConditionsTypeAMemberArray;

}

#endif //OSC_END_CONDITIONS_TYPE_A_H
