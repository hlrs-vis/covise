/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_CONDITION_BASE_H
#define OSC_CONDITION_BASE_H

#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>

#include <oscSimulationTime.h>
#include <oscTimeOfDay.h>
#include <oscReachPosition.h>
#include <oscDistance.h>
#include <oscVelocity.h>
#include <oscStandsStill.h>
#include <oscAcceleration.h>
#include <oscStoppingDistance.h>
#include <oscTimeToCollision.h>
#include <oscTimeHeadway.h>
#include <oscReferenceHanding.h>
#include <oscOffroad.h>
#include <oscCollision.h>
#include <oscNumericCondition.h>
#include <oscCommand.h>
#include <oscUserDefined.h>


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscConditionBase: public oscObjectBase
{
public:
    oscConditionBase()
    {
        OSC_OBJECT_ADD_MEMBER(simulationTime, "oscSimulationTime");
        OSC_OBJECT_ADD_MEMBER(timeOfDay, "oscTimeOfDay");
        OSC_OBJECT_ADD_MEMBER(reachPosition, "oscReachPosition");
        OSC_OBJECT_ADD_MEMBER(distance, "oscDistance");
        OSC_OBJECT_ADD_MEMBER(velocity, "oscVelocity");
        OSC_OBJECT_ADD_MEMBER(standsStill, "oscStandsStill");
        OSC_OBJECT_ADD_MEMBER(acceleration, "oscAcceleration");
        OSC_OBJECT_ADD_MEMBER(stoppingDistance, "oscStoppingDistance");
        OSC_OBJECT_ADD_MEMBER(timeToCollision, "oscTimeToCollision");
        OSC_OBJECT_ADD_MEMBER(timeHeadway, "oscTimeHeadway");
        OSC_OBJECT_ADD_MEMBER(referenceHanding, "oscReferenceHanding");
        OSC_OBJECT_ADD_MEMBER(offroad, "oscOffroad");
        OSC_OBJECT_ADD_MEMBER(collision, "oscCollision");
        OSC_OBJECT_ADD_MEMBER(numericCondition, "oscNumericCondition");
        OSC_OBJECT_ADD_MEMBER(command, "oscCommand");
        OSC_OBJECT_ADD_MEMBER(userDefined, "oscUserDefined");
    };

    oscSimulationTimeMember simulationTime;
    oscTimeOfDayMember timeOfDay;
    oscReachPositionMember reachPosition;
    oscDistanceMember distance;
    oscVelocityMember velocity;
    oscStandsStillMember standsStill;
    oscAccelerationMember acceleration;
    oscStoppingDistanceMember stoppingDistance;
    oscTimeToCollisionMember timeToCollision;
    oscTimeHeadwayMember timeHeadway;
    oscReferenceHandingMember referenceHanding;
    oscOffroadMember offroad;
    oscCollisionMember collision;
    oscNumericConditionMember numericCondition;
    oscCommandMember command;
    oscUserDefinedMember userDefined;
};

typedef oscObjectVariable<oscConditionBase *> oscConditionBaseMember;

}

#endif //OSC_CONDITION_BASE_H
