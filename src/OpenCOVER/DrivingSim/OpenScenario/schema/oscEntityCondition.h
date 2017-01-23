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
#include "oscSpeedCondition.h"
#include "oscRelativeSpeed.h"
#include "oscTraveledDistance.h"
#include "oscReachPosition.h"
#include "oscDistanceCondition.h"
#include "oscRelativeDistance.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscEntityCondition : public oscObjectBase
{
public:
oscEntityCondition()
{
        OSC_OBJECT_ADD_MEMBER(EndOfRoad, "oscEndOfRoad");
        OSC_OBJECT_ADD_MEMBER(Collision, "oscCollision");
        OSC_OBJECT_ADD_MEMBER(Offroad, "oscOffroad");
        OSC_OBJECT_ADD_MEMBER(TimeHeadway, "oscTimeHeadway");
        OSC_OBJECT_ADD_MEMBER(TimeToCollision, "oscTimeToCollision");
        OSC_OBJECT_ADD_MEMBER(Acceleration, "oscAcceleration");
        OSC_OBJECT_ADD_MEMBER(StandStill, "oscStandStill");
        OSC_OBJECT_ADD_MEMBER(SpeedCondition, "oscSpeedCondition");
        OSC_OBJECT_ADD_MEMBER(RelativeSpeed, "oscRelativeSpeed");
        OSC_OBJECT_ADD_MEMBER(TraveledDistance, "oscTraveledDistance");
        OSC_OBJECT_ADD_MEMBER(ReachPosition, "oscReachPosition");
        OSC_OBJECT_ADD_MEMBER(DistanceCondition, "oscDistanceCondition");
        OSC_OBJECT_ADD_MEMBER(RelativeDistance, "oscRelativeDistance");
    };
    oscEndOfRoadMember EndOfRoad;
    oscCollisionMember Collision;
    oscOffroadMember Offroad;
    oscTimeHeadwayMember TimeHeadway;
    oscTimeToCollisionMember TimeToCollision;
    oscAccelerationMember Acceleration;
    oscStandStillMember StandStill;
    oscSpeedConditionMember SpeedCondition;
    oscRelativeSpeedMember RelativeSpeed;
    oscTraveledDistanceMember TraveledDistance;
    oscReachPositionMember ReachPosition;
    oscDistanceConditionMember DistanceCondition;
    oscRelativeDistanceMember RelativeDistance;

};

typedef oscObjectVariable<oscEntityCondition *> oscEntityConditionMember;
typedef oscObjectVariableArray<oscEntityCondition *> oscEntityConditionArrayMember;


}

#endif //OSCENTITYCONDITION_H
