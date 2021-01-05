/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCCONDITIONDISTANCE_H
#define OSCCONDITIONDISTANCE_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscTimeHeadway.h"
#include "oscPosition.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscConditionDistance : public oscObjectBase
{
public:
oscConditionDistance()
{
        OSC_ADD_MEMBER(value, 0);
        OSC_ADD_MEMBER(freespace, 0);
        OSC_ADD_MEMBER(alongRoute, 0);
        OSC_ADD_MEMBER(rule, 0);
        OSC_OBJECT_ADD_MEMBER(Position, "oscPosition", 0);
        rule.enumType = Enum_ruleType::instance();
    };
        const char *getScope(){return "/OSCCondition/ByEntity/EntityCondition";};
    oscDouble value;
    oscBool freespace;
    oscBool alongRoute;
    oscEnum rule;
    oscPositionMember Position;

};

typedef oscObjectVariable<oscConditionDistance *> oscConditionDistanceMember;
typedef oscObjectVariableArray<oscConditionDistance *> oscConditionDistanceArrayMember;


}

#endif //OSCCONDITIONDISTANCE_H
