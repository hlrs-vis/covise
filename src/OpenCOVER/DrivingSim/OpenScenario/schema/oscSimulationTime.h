/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCSIMULATIONTIME_H
#define OSCSIMULATIONTIME_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"
#include "oscObjectVariableArray.h"

#include "oscVariables.h"
#include "schema/oscTimeHeadway.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscSimulationTime : public oscObjectBase
{
public:
oscSimulationTime()
{
        OSC_ADD_MEMBER(value);
        OSC_ADD_MEMBER(rule);
        rule.enumType = Enum_ruleType::instance();
    };
    oscDouble value;
    oscEnum rule;

};

typedef oscObjectVariable<oscSimulationTime *> oscSimulationTimeMember;
typedef oscObjectVariableArray<oscSimulationTime *> oscSimulationTimeArrayMember;


}

#endif //OSCSIMULATIONTIME_H
