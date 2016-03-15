/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_CONDITION_BASE_H
#define OSC_CONDITION_BASE_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscSimulationTime.h"
#include "oscTimeOfDay.h"
#include "oscReachPosition.h"
#include "oscDistance.h"
#include "oscConditionChoiceTypeA.h"
#include "oscStandsStill.h"
#include "oscStoppingDistance.h"
#include "oscTimeToCollision.h"
#include "oscTimeHeadway.h"
#include "oscReferenceHandling.h"
#include "oscOffroad.h"
#include "oscCollision.h"
#include "oscNumericCondition.h"
#include "oscCommand.h"
#include "oscAfterManeuvers.h"
#include "oscEndOfRoad.h"
#include "oscUserDefined.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscConditionBase: public oscObjectBase
{
public:
    oscConditionBase()
    {
        OSC_OBJECT_ADD_MEMBER_CHOICE(simulationTime, "oscSimulationTime");
        OSC_OBJECT_ADD_MEMBER_CHOICE(timeOfDay, "oscTimeOfDay");
        OSC_OBJECT_ADD_MEMBER_CHOICE(reachPosition, "oscReachPosition");
        OSC_OBJECT_ADD_MEMBER_CHOICE(distance, "oscDistance");
        OSC_OBJECT_ADD_MEMBER_CHOICE(velocity, "oscConditionChoiceTypeA");
        OSC_OBJECT_ADD_MEMBER_CHOICE(standsStill, "oscStandsStill");
        OSC_OBJECT_ADD_MEMBER_CHOICE(acceleration, "oscConditionChoiceTypeA");
        OSC_OBJECT_ADD_MEMBER_CHOICE(stoppingDistance, "oscStoppingDistance");
        OSC_OBJECT_ADD_MEMBER_CHOICE(timeToCollision, "oscTimeToCollision");
        OSC_OBJECT_ADD_MEMBER_CHOICE(timeHeadway, "oscTimeHeadway");
        OSC_OBJECT_ADD_MEMBER_CHOICE(referenceHandling, "oscReferenceHandling");
        OSC_OBJECT_ADD_MEMBER_CHOICE(offroad, "oscOffroad");
        OSC_OBJECT_ADD_MEMBER_CHOICE(collision, "oscCollision");
        OSC_OBJECT_ADD_MEMBER_CHOICE(numericCondition, "oscNumericCondition");
        OSC_OBJECT_ADD_MEMBER_CHOICE(command, "oscCommand");
        OSC_OBJECT_ADD_MEMBER_CHOICE(afterManeuvers, "oscAfterManeuvers");
        OSC_OBJECT_ADD_MEMBER_CHOICE(endOfRoad, "oscEndOfRoad");
        OSC_OBJECT_ADD_MEMBER_CHOICE(userDefined, "oscUserDefined");
    };

    oscSimulationTimeMember simulationTime;
    oscTimeOfDayMember timeOfDay;
    oscReachPositionMember reachPosition;
    oscDistanceMember distance;
    oscConditionChoiceTypeAMember velocity;
    oscStandsStillMember standsStill;
    oscConditionChoiceTypeAMember acceleration;
    oscStoppingDistanceMember stoppingDistance;
    oscTimeToCollisionMember timeToCollision;
    oscTimeHeadwayMember timeHeadway;
    oscReferenceHandlingMember referenceHandling;
    oscOffroadMember offroad;
    oscCollisionMember collision;
    oscNumericConditionMember numericCondition;
    oscCommandMember command;
    oscAfterManeuversMember afterManeuvers;
    oscEndOfRoadMember endOfRoad;
    oscUserDefinedMember userDefined;
};

typedef oscObjectVariable<oscConditionBase *> oscConditionBaseMember;

}

#endif //OSC_CONDITION_BASE_H
