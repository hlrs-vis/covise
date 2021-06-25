/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCENTITYCONDITION_H
#define OSCENTITYCONDITION_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscEndOfRoad.h"
#include "oscCollision.h"
#include "oscOffroad.h"
#include "oscTimeHeadway.h"
#include "oscTimeToCollision.h"
#include "oscAcceleration.h"
#include "oscStandStill.h"
#include "oscConditionSpeed.h"
#include "oscRelativeSpeed.h"
#include "oscTraveledDistance.h"
#include "oscReachPosition.h"
#include "oscConditionDistance.h"
#include "oscRelativeDistance.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscEntityCondition : public oscObjectBase
{
public:
oscEntityCondition()
{
        OSC_OBJECT_ADD_MEMBER(EndOfRoad, "oscEndOfRoad", 1);
        OSC_OBJECT_ADD_MEMBER(Collision, "oscCollision", 1);
        OSC_OBJECT_ADD_MEMBER(Offroad, "oscOffroad", 1);
        OSC_OBJECT_ADD_MEMBER(TimeHeadway, "oscTimeHeadway", 1);
        OSC_OBJECT_ADD_MEMBER(TimeToCollision, "oscTimeToCollision", 1);
        OSC_OBJECT_ADD_MEMBER(Acceleration, "oscAcceleration", 1);
        OSC_OBJECT_ADD_MEMBER(StandStill, "oscStandStill", 1);
        OSC_OBJECT_ADD_MEMBER(Speed, "oscConditionSpeed", 1);
        OSC_OBJECT_ADD_MEMBER(RelativeSpeed, "oscRelativeSpeed", 1);
        OSC_OBJECT_ADD_MEMBER(TraveledDistance, "oscTraveledDistance", 1);
        OSC_OBJECT_ADD_MEMBER(ReachPosition, "oscReachPosition", 1);
        OSC_OBJECT_ADD_MEMBER(Distance, "oscConditionDistance", 1);
        OSC_OBJECT_ADD_MEMBER(RelativeDistance, "oscRelativeDistance", 1);
    };
        const char *getScope(){return "/OSCCondition/ByEntity";};
    oscEndOfRoadMember EndOfRoad;
    oscCollisionMember Collision;
    oscOffroadMember Offroad;
    oscTimeHeadwayMember TimeHeadway;
    oscTimeToCollisionMember TimeToCollision;
    oscAccelerationMember Acceleration;
    oscStandStillMember StandStill;
    oscConditionSpeedMember Speed;
    oscRelativeSpeedMember RelativeSpeed;
    oscTraveledDistanceMember TraveledDistance;
    oscReachPositionMember ReachPosition;
    oscConditionDistanceMember Distance;
    oscRelativeDistanceMember RelativeDistance;

};

typedef oscObjectVariable<oscEntityCondition *> oscEntityConditionMember;
typedef oscObjectVariableArray<oscEntityCondition *> oscEntityConditionArrayMember;


}

#endif //OSCENTITYCONDITION_H
