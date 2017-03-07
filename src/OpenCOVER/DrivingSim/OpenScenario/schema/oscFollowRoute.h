/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCFOLLOWROUTE_H
#define OSCFOLLOWROUTE_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscRoute.h"
#include "oscCatalogReference.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscFollowRoute : public oscObjectBase
{
public:
oscFollowRoute()
{
        OSC_OBJECT_ADD_MEMBER(Route, "oscRoute", 1);
        OSC_OBJECT_ADD_MEMBER(CatalogReference, "oscCatalogReference", 1);
    };
        const char *getScope(){return "/OSCPrivateAction/Routing";};
    oscRouteMember Route;
    oscCatalogReferenceMember CatalogReference;

};

typedef oscObjectVariable<oscFollowRoute *> oscFollowRouteMember;
typedef oscObjectVariableArray<oscFollowRoute *> oscFollowRouteArrayMember;


}

#endif //OSCFOLLOWROUTE_H
