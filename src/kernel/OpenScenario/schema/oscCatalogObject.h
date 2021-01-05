/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCCATALOGOBJECT_H
#define OSCCATALOGOBJECT_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscVehicle.h"
#include "oscDriver.h"
#include "oscPedestrian.h"
#include "oscPedestrianController.h"
#include "oscMiscObject.h"
#include "oscEnvironment.h"
#include "oscManeuver.h"
#include "oscTrajectory.h"
#include "oscRoute.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscCatalogObject : public oscObjectBase
{
public:
oscCatalogObject()
{
        OSC_ADD_MEMBER(name, 0);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Vehicle, "oscVehicle", 0);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Driver, "oscDriver", 0);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Pedestrian, "oscPedestrian", 0);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(PedestrianController, "oscPedestrianController", 0);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(MiscObject, "oscMiscObject", 0);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Environment, "oscEnvironment", 0);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Maneuver, "oscManeuver", 0);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Trajectory, "oscTrajectory", 0);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Route, "oscRoute", 0);
    };
        const char *getScope(){return "/Catalog/CatalogOpenSCENARIO";};
    oscString name;
    oscVehicleArrayMember Vehicle;
    oscDriverArrayMember Driver;
    oscPedestrianArrayMember Pedestrian;
    oscPedestrianControllerArrayMember PedestrianController;
    oscMiscObjectArrayMember MiscObject;
    oscEnvironmentArrayMember Environment;
    oscManeuverArrayMember Maneuver;
    oscTrajectoryArrayMember Trajectory;
    oscRouteArrayMember Route;

};

typedef oscObjectVariable<oscCatalogObject *> oscCatalogObjectMember;
typedef oscObjectVariableArray<oscCatalogObject *> oscCatalogObjectArrayMember;


}

#endif //OSCCATALOGOBJECT_H
