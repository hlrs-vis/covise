/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_CANCEL_CONDITIONS_TYPE_B_H
#define OSC_CANCEL_CONDITIONS_TYPE_B_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectArrayVariable.h"

#include "oscConditionTypeB.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscCancelConditionsTypeB: public oscObjectBase
{
public:
    oscCancelConditionsTypeB()
    {
        OSC_OBJECT_ADD_MEMBER(cancelCondition, "oscConditionTypeB");
    };

    oscConditionTypeBMember cancelCondition;
};

typedef oscObjectArrayVariable<oscCancelConditionsTypeB *> oscCancelConditionsTypeBArrayMember;

}

#endif //OSC_CANCEL_CONDITIONS_TYPE_B_H
