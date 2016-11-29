/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCCATALOGS_H
#define OSCCATALOGS_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscVehicleCatalog.h"
#include "schema/oscDriverCatalog.h"
#include "schema/oscPedestrianCatalog.h"
#include "schema/oscPedestrianControllerCatalog.h"
#include "schema/oscMiscObjectCatalog.h"
#include "schema/oscEnvironmentCatalog.h"
#include "schema/oscManeuverCatalog.h"
#include "schema/oscTrajectoryCatalog.h"
#include "schema/oscRouteCatalog.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscCatalogs : public oscObjectBase
{
public:
    oscCatalogs()
    {
        OSC_OBJECT_ADD_MEMBER(VehicleCatalog, "oscVehicleCatalog");
        OSC_OBJECT_ADD_MEMBER(DriverCatalog, "oscDriverCatalog");
        OSC_OBJECT_ADD_MEMBER(PedestrianCatalog, "oscPedestrianCatalog");
        OSC_OBJECT_ADD_MEMBER(PedestrianControllerCatalog, "oscPedestrianControllerCatalog");
        OSC_OBJECT_ADD_MEMBER(MiscObjectCatalog, "oscMiscObjectCatalog");
        OSC_OBJECT_ADD_MEMBER(EnvironmentCatalog, "oscEnvironmentCatalog");
        OSC_OBJECT_ADD_MEMBER(ManeuverCatalog, "oscManeuverCatalog");
        OSC_OBJECT_ADD_MEMBER(TrajectoryCatalog, "oscTrajectoryCatalog");
        OSC_OBJECT_ADD_MEMBER(RouteCatalog, "oscRouteCatalog");
    };
    oscVehicleCatalogMember VehicleCatalog;
    oscDriverCatalogMember DriverCatalog;
    oscPedestrianCatalogMember PedestrianCatalog;
    oscPedestrianControllerCatalogMember PedestrianControllerCatalog;
    oscMiscObjectCatalogMember MiscObjectCatalog;
    oscEnvironmentCatalogMember EnvironmentCatalog;
    oscManeuverCatalogMember ManeuverCatalog;
    oscTrajectoryCatalogMember TrajectoryCatalog;
    oscRouteCatalogMember RouteCatalog;

};

typedef oscObjectVariable<oscCatalogs *> oscCatalogsMember;


}

#endif //OSCCATALOGS_H
