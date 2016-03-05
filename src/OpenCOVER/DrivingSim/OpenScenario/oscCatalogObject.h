/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_CATALOG_OBJECT_H
#define OSC_CATALOG_OBJECT_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscCatalogBase.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscCatalogObject: public oscObjectBase
{
public:
    oscCatalogObject()
    {
        OSC_OBJECT_ADD_MEMBER(vehicleCatalog, "oscCatalogBase");
        OSC_OBJECT_ADD_MEMBER(driverCatalog, "oscCatalogBase");
        OSC_OBJECT_ADD_MEMBER(observerCatalog, "oscCatalogBase");
        OSC_OBJECT_ADD_MEMBER(pedestrianCatalog, "oscCatalogBase");
        OSC_OBJECT_ADD_MEMBER(miscObjectCatalog, "oscCatalogBase");
    };

    oscCatalogBaseMemberCatalog vehicleCatalog;
    oscCatalogBaseMemberCatalog driverCatalog;
    oscCatalogBaseMemberCatalog observerCatalog;
    oscCatalogBaseMemberCatalog pedestrianCatalog;
    oscCatalogBaseMemberCatalog miscObjectCatalog;
};

typedef oscObjectVariable<oscCatalogObject *> oscCatalogObjectMember;

}

#endif //OSC_CATALOG_OBJECT_H
