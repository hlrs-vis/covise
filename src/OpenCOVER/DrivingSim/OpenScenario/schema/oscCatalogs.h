/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCCATALOGS_H
#define OSCCATALOGS_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscVehicleCatalog.h"
#include "oscDriverCatalog.h"
#include "oscPedestrianCatalog.h"
#include "oscPedestrianControllerCatalog.h"
#include "oscMiscObjectCatalog.h"
#include "oscEnvironmentCatalog.h"
#include "oscManeuverCatalog.h"
#include "oscTrajectoryCatalog.h"
#include "oscRouteCatalog.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscCatalogs : public oscObjectBase
{
public:
oscCatalogs()
{
        OSC_OBJECT_ADD_MEMBER(VehicleCatalog, "oscVehicleCatalog", 0);
        OSC_OBJECT_ADD_MEMBER(DriverCatalog, "oscDriverCatalog", 0);
        OSC_OBJECT_ADD_MEMBER(PedestrianCatalog, "oscPedestrianCatalog", 0);
        OSC_OBJECT_ADD_MEMBER(PedestrianControllerCatalog, "oscPedestrianControllerCatalog", 0);
        OSC_OBJECT_ADD_MEMBER(MiscObjectCatalog, "oscMiscObjectCatalog", 0);
        OSC_OBJECT_ADD_MEMBER(EnvironmentCatalog, "oscEnvironmentCatalog", 0);
        OSC_OBJECT_ADD_MEMBER(ManeuverCatalog, "oscManeuverCatalog", 0);
        OSC_OBJECT_ADD_MEMBER(TrajectoryCatalog, "oscTrajectoryCatalog", 0);
        OSC_OBJECT_ADD_MEMBER(RouteCatalog, "oscRouteCatalog", 0);
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
typedef oscObjectVariableArray<oscCatalogs *> oscCatalogsArrayMember;


}

#endif //OSCCATALOGS_H
