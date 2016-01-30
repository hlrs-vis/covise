/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_OBJECT_CATALOG_H
#define OSC_OBJECT_CATALOG_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVehicleCatalog.h"
#include "oscDriverCatalog.h"
#include "oscObserverCatalog.h"
#include "oscPedestrianCatalog.h"
#include "oscMiscObjectCatalog.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscObjectCatalog: public oscObjectBase
{
public:
    oscObjectCatalog()
    {
        OSC_OBJECT_ADD_MEMBER(vehicleCatalog, "oscVehicleCatalog");
        OSC_OBJECT_ADD_MEMBER(driverCatalog, "oscDriverCatalog");
        OSC_OBJECT_ADD_MEMBER(observerCatalog, "oscObserverCatalog");
        OSC_OBJECT_ADD_MEMBER(pedestrianCatalog, "oscPedestrianCatalog");
        OSC_OBJECT_ADD_MEMBER(miscObjectCatalog, "oscMiscObjectCatalog");
    };

    oscVehicleCatalogMember vehicleCatalog;
    oscDriverCatalogMember driverCatalog;
    oscObserverCatalogMember observerCatalog;
    oscPedestrianCatalogMember pedestrianCatalog;
    oscMiscObjectCatalogMember miscObjectCatalog;
};

typedef oscObjectVariable<oscObjectCatalog *> oscObjectCatalogMember;

}

#endif //OSC_OBJECT_CATALOG_H
