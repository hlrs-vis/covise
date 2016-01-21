/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_CANCEL_CONDITION_TYPE_A_H
#define OSC_CANCEL_CONDITION_TYPE_A_H

#include "oscExport.h"
#include "oscConditionTypeA.h"
#include "oscObjectVariable.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscCancelConditionTypeA: public oscConditionTypeA
{
public:
    oscCancelConditionTypeA()
    {

    };

};

typedef oscObjectVariable<oscCancelConditionTypeA *> oscCancelConditionTypeAMember;

}

#endif //OSC_CANCEL_CONDITION_TYPE_A_H
