/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCTIMEOFDAY_H
#define OSCTIMEOFDAY_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscTimeHeadway.h"
#include "oscTime.h"
#include "oscDate.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscTimeOfDay : public oscObjectBase
{
public:
oscTimeOfDay()
{
        OSC_ADD_MEMBER(animation);
        OSC_ADD_MEMBER(rule);
        OSC_OBJECT_ADD_MEMBER(Time, "oscTime");
        OSC_OBJECT_ADD_MEMBER(Date, "oscDate");
        rule.enumType = Enum_ruleType::instance();
    };
    oscBool animation;
    oscEnum rule;
    oscTimeMember Time;
    oscDateMember Date;

};

typedef oscObjectVariable<oscTimeOfDay *> oscTimeOfDayMember;
typedef oscObjectVariableArray<oscTimeOfDay *> oscTimeOfDayArrayMember;


}

#endif //OSCTIMEOFDAY_H
