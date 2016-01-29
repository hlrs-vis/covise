/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_TEST_H
#define OSC_TEST_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscMiscObject.h"
#include "oscPedestrian.h"
#include "oscObserverTypeA.h"
#include "oscEntity.h"
#include "oscRouting.h"
#include "oscDriver.h"
#include "oscEnvironment.h"
#include "oscVehicle.h"
#include "oscManeuverTypeA.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscTest: public oscObjectBase
{
public:
    oscTest()
    {
        OSC_OBJECT_ADD_MEMBER(miscObject, "oscMiscObject");
        OSC_OBJECT_ADD_MEMBER(pedestrian, "oscPedestrian");
        OSC_OBJECT_ADD_MEMBER(observer, "oscObserverTypeA");
        OSC_OBJECT_ADD_MEMBER(entity, "oscEntity");
        OSC_OBJECT_ADD_MEMBER(routing, "oscRouting");
        OSC_OBJECT_ADD_MEMBER(driver, "oscDriver");
        OSC_OBJECT_ADD_MEMBER(environment, "oscEnvironment");
        OSC_OBJECT_ADD_MEMBER(vehicle, "oscVehicle");
        OSC_OBJECT_ADD_MEMBER(maneuver, "oscManeuverTypeA");
    };

    oscMiscObjectMember miscObject;
    oscPedestrianMember pedestrian;
    oscObserverTypeAMember observer;
    oscEntityMember entity;
    oscRoutingMember routing;
    oscDriverMember driver;
    oscEnvironmentMember environment;
    oscVehicleMember vehicle;
    oscManeuverTypeAMember maneuver;
};

typedef oscObjectVariable<oscTest *> oscTestMember;

}

#endif //OSC_TEST_H
