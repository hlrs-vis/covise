/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_CATALOGS_H
#define OSC_CATALOGS_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscObjectCatalog.h"
#include "oscEntityCatalog.h"
#include "oscEnvironmentCatalog.h"
#include "oscManeuverCatalog.h"
#include "oscRoutingCatalog.h"
#include "oscUserDataList.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscCatalogs: public oscObjectBase
{
public:
    oscCatalogs()
    {
        OSC_OBJECT_ADD_MEMBER(objectCatalog, "oscObjectCatalog");
        OSC_OBJECT_ADD_MEMBER(entityCatalog, "oscEntityCatalog");
        OSC_OBJECT_ADD_MEMBER(environmentCatalog, "oscEnvironmentCatalog");
        OSC_OBJECT_ADD_MEMBER(maneuverCatalog, "oscManeuverCatalog");
        OSC_OBJECT_ADD_MEMBER(routingCatalog, "oscRoutingCatalog");
        OSC_OBJECT_ADD_MEMBER(userDataList, "oscUserDataList");
    };

    oscObjectCatalogMember objectCatalog;
    oscEntityCatalogMember entityCatalog;
    oscEnvironmentCatalogMember environmentCatalog;
    oscManeuverCatalogMember maneuverCatalog;
    oscRoutingCatalogMember routingCatalog;
    oscUserDataListArrayMember userDataList;
};

typedef oscObjectVariable<oscCatalogs *> oscCatalogsMember;

}

#endif //OSC_CATALOGS_H
