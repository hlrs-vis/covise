/* This RoutingCatalog is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_ROUTING_CATALOG_H
#define OSC_ROUTING_CATALOG_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>
#include <oscCatalog.h>

namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscRoutingCatalog: public oscObjectBase
{
public:
    oscRoutingCatalog()
    {
        OSC_OBJECT_ADD_MEMBER(routing,"oscCatalog");
    };
    oscCatalogMember routing;
};

typedef oscObjectVariable<oscRoutingCatalog *> oscRoutingCatalogMember;

}

#endif //OSC_ROUTING_CATALOG_HS
