/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_CATALOGS_H
#define OSC_CATALOGS_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscCatalogObject.h"
#include "oscCatalogBase.h"
#include "oscUserDataList.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscCatalogs: public oscObjectBase
{
public:
    oscCatalogs()
    {
        OSC_OBJECT_ADD_MEMBER(objectCatalog, "oscCatalogObject");
        OSC_OBJECT_ADD_MEMBER(entityCatalog, "oscCatalogBase");
        OSC_OBJECT_ADD_MEMBER(environmentCatalog, "oscCatalogBase");
        OSC_OBJECT_ADD_MEMBER(maneuverCatalog, "oscCatalogBase");
        OSC_OBJECT_ADD_MEMBER(routingCatalog, "oscCatalogBase");
        OSC_OBJECT_ADD_MEMBER(userDataList, "oscUserDataList");
    };

    oscCatalogObjectMember objectCatalog;
    oscCatalogBaseMember entityCatalog;
    oscCatalogBaseMember environmentCatalog;
    oscCatalogBaseMember maneuverCatalog;
    oscCatalogBaseMember routingCatalog;
    oscUserDataListMemberArray userDataList;
};

typedef oscObjectVariable<oscCatalogs *> oscCatalogsMember;

}

#endif //OSC_CATALOGS_H
