/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCTIMETOCOLLISION_H
#define OSCTIMETOCOLLISION_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscTimeHeadway.h"
#include "oscCollisionTarget.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscTimeToCollision : public oscObjectBase
{
public:
oscTimeToCollision()
{
        OSC_ADD_MEMBER(value, 0);
        OSC_ADD_MEMBER(freespace, 0);
        OSC_ADD_MEMBER(alongRoute, 0);
        OSC_ADD_MEMBER(rule, 0);
        OSC_OBJECT_ADD_MEMBER(CollisionTarget, "oscCollisionTarget", 0);
        rule.enumType = Enum_ruleType::instance();
    };
    oscDouble value;
    oscBool freespace;
    oscBool alongRoute;
    oscEnum rule;
    oscCollisionTargetMember CollisionTarget;

};

typedef oscObjectVariable<oscTimeToCollision *> oscTimeToCollisionMember;
typedef oscObjectVariableArray<oscTimeToCollision *> oscTimeToCollisionArrayMember;


}

#endif //OSCTIMETOCOLLISION_H
