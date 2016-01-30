/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_START_CONDITION_TYPE_C_H
#define OSC_START_CONDITION_TYPE_C_H

#include "oscExport.h"
#include "oscNamedObject.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "oscConditionBase.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscStartConditionTypeC: public oscNamedObject
{
public:
    oscStartConditionTypeC()
    {
        OSC_ADD_MEMBER(delayTime);
        OSC_OBJECT_ADD_MEMBER(condition, "oscConditionBase");
    };

    oscDouble delayTime;
    oscConditionBaseMember condition;
};

typedef oscObjectVariable<oscStartConditionTypeC *> oscStartConditionTypeCMember;

}

#endif //OSC_START_CONDITION_TYPE_C_H
