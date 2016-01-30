/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_CANCEL_CONDITIONS_GROUP_TYPE_A_H
#define OSC_CANCEL_CONDITIONS_GROUP_TYPE_A_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscCancelConditionsTypeA.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscCancelConditionsGroupTypeA: public oscObjectBase
{
public:
    oscCancelConditionsGroupTypeA()
    {
        OSC_OBJECT_ADD_MEMBER(cancelConditions, "oscCancelConditionsTypeA");
    };

    oscCancelConditionsTypeAArrayMember cancelConditions;
};

typedef oscObjectVariable<oscCancelConditionsGroupTypeA *> oscCancelConditionsGroupTypeAMember;

}

#endif //OSC_CANCEL_CONDITIONS_GROUP_TYPE_A_H
