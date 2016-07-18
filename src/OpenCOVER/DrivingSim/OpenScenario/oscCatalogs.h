/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_CATALOGS_H
#define OSC_CATALOGS_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscObjectCatalogs.h"
#include "oscCatalog.h"
#include "oscUserDataList.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscCatalogs: public oscObjectBase
{
public:
    oscCatalogs()
    {
        OSC_OBJECT_ADD_MEMBER(objectCatalogs, "oscObjectCatalogs");
        OSC_OBJECT_ADD_MEMBER(entityCatalog, "oscCatalog");
        OSC_OBJECT_ADD_MEMBER(environmentCatalog, "oscCatalog");
        OSC_OBJECT_ADD_MEMBER(maneuverCatalog, "oscCatalog");
        OSC_OBJECT_ADD_MEMBER(routingCatalog, "oscCatalog");
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(userDataList, "oscUserDataList");
    };

    oscObjectCatalogsMember objectCatalogs;
    oscCatalogMember entityCatalog;
    oscCatalogMember environmentCatalog;
    oscCatalogMember maneuverCatalog;
    oscCatalogMember routingCatalog;
    oscUserDataListMemberArray userDataList;
};

typedef oscObjectVariable<oscCatalogs *> oscCatalogsMember;

}

#endif //OSC_CATALOGS_H
