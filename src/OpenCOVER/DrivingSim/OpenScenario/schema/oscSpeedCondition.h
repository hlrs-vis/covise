/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCSPEEDCONDITION_H
#define OSCSPEEDCONDITION_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscTimeHeadway.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscSpeedCondition : public oscObjectBase
{
public:
    oscSpeedCondition()
    {
        OSC_ADD_MEMBER(value);
        OSC_ADD_MEMBER(rule);
    };
    oscDouble value;
    oscEnum rule;

};

typedef oscObjectVariable<oscSpeedCondition *> oscSpeedConditionMember;


}

#endif //OSCSPEEDCONDITION_H
