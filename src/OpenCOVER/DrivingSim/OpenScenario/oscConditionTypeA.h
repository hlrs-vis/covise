/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_CONDITION_TYPE_A_H
#define OSC_CONDITION_TYPE_A_H

#include "oscExport.h"
#include "oscNamedObject.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "oscConditionBase.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscConditionTypeA: public oscNamedObject
{
public:
    oscConditionTypeA()
    {
        OSC_ADD_MEMBER(counter);
        OSC_OBJECT_ADD_MEMBER(condition, "oscConditionBase");
    };

    oscInt counter;
    oscConditionBaseMember condition;
};

typedef oscObjectVariable<oscConditionTypeA *> oscConditionTypeAMember;

}

#endif //OSC_CONDITION_TYPE_A_H
