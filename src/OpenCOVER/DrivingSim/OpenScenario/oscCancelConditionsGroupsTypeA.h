/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_CANCEL_CONDITIONS_GROUPS_TYPE_A_H
#define OSC_CANCEL_CONDITIONS_GROUPS_TYPE_A_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectArrayVariable.h"

#include "oscCancelConditionsGroupTypeA.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscCancelConditionsGroupsTypeA: public oscObjectBase
{
public:
    oscCancelConditionsGroupsTypeA()
    {
        OSC_OBJECT_ADD_MEMBER(cancelConditionGroup, "oscCancelConditionsGroupTypeA");
    };

    oscCancelConditionsGroupTypeAMember cancelConditionGroup;
};

typedef oscObjectArrayVariable<oscCancelConditionsGroupsTypeA *> oscCancelConditionsGroupsTypeAArrayMember;

}

#endif /* OSC_CANCEL_CONDITIONS_GROUPS_TYPE_A_H */
