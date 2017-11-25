/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCCOLLISIONTARGET_H
#define OSCCOLLISIONTARGET_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscPosition.h"
#include "oscEntity.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscCollisionTarget : public oscObjectBase
{
public:
oscCollisionTarget()
{
        OSC_OBJECT_ADD_MEMBER(Position, "oscPosition", 1);
        OSC_OBJECT_ADD_MEMBER(Entity, "oscEntity", 1);
    };
        const char *getScope(){return "/OSCCondition/ByEntity/EntityCondition/TimeToCollision";};
    oscPositionMember Position;
    oscEntityMember Entity;

};

typedef oscObjectVariable<oscCollisionTarget *> oscCollisionTargetMember;
typedef oscObjectVariableArray<oscCollisionTarget *> oscCollisionTargetArrayMember;


}

#endif //OSCCOLLISIONTARGET_H
