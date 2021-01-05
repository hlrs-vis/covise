/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_TEST_H
#define OSC_TEST_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "schema/oscPedestrian.h"
#include "schema/oscEntity.h"
#include "schema/oscRouting.h"
#include "schema/oscDriver.h"
#include "schema/oscEnvironment.h"
#include "schema/oscVehicle.h"
#include "schema/oscManeuver.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscTest: public oscObjectBase
{
public:
    oscTest()
    {
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(pedestrian, "oscPedestrian", 0);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(entity, "oscEntity", 0);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(routing, "oscRouting", 0);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(driver, "oscDriver", 0);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(environment, "oscEnvironment", 0);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(vehicle, "oscVehicle", 0);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(maneuver, "oscManeuver", 0);
    };

    oscPedestrianMember pedestrian;
    oscEntityMember entity;
    oscRoutingMember routing;
    oscDriverMember driver;
    oscEnvironmentMember environment;
    oscVehicleMember vehicle;
    oscManeuverMember maneuver;
};

typedef oscObjectVariable<oscTest *> oscTestMember;

}

#endif //OSC_TEST_H
