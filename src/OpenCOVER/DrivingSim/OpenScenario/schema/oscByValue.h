/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCBYVALUE_H
#define OSCBYVALUE_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscConditionParameter.h"
#include "oscTimeOfDay.h"
#include "oscSimulationTime.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscByValue : public oscObjectBase
{
public:
oscByValue()
{
        OSC_OBJECT_ADD_MEMBER(ConditionParameter, "oscConditionParameter", 1);
        OSC_OBJECT_ADD_MEMBER(TimeOfDay, "oscTimeOfDay", 1);
        OSC_OBJECT_ADD_MEMBER(SimulationTime, "oscSimulationTime", 1);
    };
    oscConditionParameterMember ConditionParameter;
    oscTimeOfDayMember TimeOfDay;
    oscSimulationTimeMember SimulationTime;

};

typedef oscObjectVariable<oscByValue *> oscByValueMember;
typedef oscObjectVariableArray<oscByValue *> oscByValueArrayMember;


}

#endif //OSCBYVALUE_H
