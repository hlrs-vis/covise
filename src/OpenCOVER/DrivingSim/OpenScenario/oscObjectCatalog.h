/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_CATALOG_OBJECT_H
#define OSC_CATALOG_OBJECT_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscCatalog.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscObjectCatalog: public oscObjectBase
{
public:
    oscObjectCatalog()
    {
        OSC_OBJECT_ADD_MEMBER(vehicleCatalog, "oscCatalog");
        OSC_OBJECT_ADD_MEMBER(driverCatalog, "oscCatalog");
        OSC_OBJECT_ADD_MEMBER(observerCatalog, "oscCatalog");
        OSC_OBJECT_ADD_MEMBER(pedestrianCatalog, "oscCatalog");
        OSC_OBJECT_ADD_MEMBER(miscObjectCatalog, "oscCatalog");
    };

    oscCatalogMember vehicleCatalog;
    oscCatalogMember driverCatalog;
    oscCatalogMember observerCatalog;
    oscCatalogMember pedestrianCatalog;
    oscCatalogMember miscObjectCatalog;
};

typedef oscObjectVariable<oscObjectCatalog *> oscObjectCatalogMember;

}

#endif //OSC_CATALOG_OBJECT_H
