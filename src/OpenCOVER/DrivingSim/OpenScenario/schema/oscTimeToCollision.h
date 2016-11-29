/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCTIMETOCOLLISION_H
#define OSCTIMETOCOLLISION_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscTimeHeadway.h"
#include "schema/oscCollisionTarget.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscTimeToCollision : public oscObjectBase
{
public:
    oscTimeToCollision()
    {
        OSC_ADD_MEMBER(value);
        OSC_ADD_MEMBER(freespace);
        OSC_ADD_MEMBER(alongRoute);
        OSC_ADD_MEMBER(rule);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(CollisionTarget, "oscCollisionTarget");
    };
    oscDouble value;
    oscBool freespace;
    oscBool alongRoute;
    oscEnum rule;
    oscCollisionTargetMember CollisionTarget;

};

typedef oscObjectVariable<oscTimeToCollision *> oscTimeToCollisionMember;


}

#endif //OSCTIMETOCOLLISION_H
