/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_TEST_H
#define OSC_TEST_H

#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>

#include <oscDriver.h>
#include <oscConditionBase.h>
#include <oscPosition.h>
#include <oscEntity.h>
#include <oscVehicle.h>
#include <oscRouting.h>
#include <oscPedestrian.h>
#include <oscVelocity.h>
#include <oscRelativePositionWorld.h>
#include <oscObserver.h>
#include <oscManeuverTypeA.h>
#include <oscMiscObject.h>
#include <oscEnvironment.h>


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscTest: public oscObjectBase
{
public:
    oscTest()
    {
        OSC_OBJECT_ADD_MEMBER(driver, "oscDriver");
        OSC_OBJECT_ADD_MEMBER(condition, "oscConditionBase");
        OSC_OBJECT_ADD_MEMBER(position, "oscPosition");
        OSC_OBJECT_ADD_MEMBER(entity, "oscEntity");
        OSC_OBJECT_ADD_MEMBER(vehicle, "oscVehicle");
        OSC_OBJECT_ADD_MEMBER(routing, "oscRouting");
        OSC_OBJECT_ADD_MEMBER(pedestrian, "oscPedestrian");
        OSC_OBJECT_ADD_MEMBER(velocity, "oscVelocity");
        OSC_OBJECT_ADD_MEMBER(relativePositionWorld, "oscRelativePositionWorld");
        OSC_OBJECT_ADD_MEMBER(observer, "oscObserver");
        OSC_OBJECT_ADD_MEMBER(maneuver, "oscManeuverTypeA");
        OSC_OBJECT_ADD_MEMBER(miscObject, "oscMiscObject");
        OSC_OBJECT_ADD_MEMBER(environment, "oscEnvironment");
    };

    oscDriverMember driver;
    oscConditionBaseMember condition;
    oscPositionMember position;
    oscEntityMember entity;
    oscVehicleMember vehicle;
    oscRoutingMember routing;
    oscPedestrianMember pedestrian;
    oscVelocityMember velocity;
    oscRelativePositionWorldMember relativePositionWorld;
    oscObserverMember observer;
    oscManeuverTypeAMember maneuver;
    oscMiscObjectMember miscObject;
    oscEnvironmentMember environment;
};

typedef oscObjectVariable<oscTest *> oscTestMember;

}

#endif //OSC_TEST_H
