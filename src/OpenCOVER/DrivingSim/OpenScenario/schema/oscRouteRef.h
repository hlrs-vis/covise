/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCROUTEREF_H
#define OSCROUTEREF_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscRoute.h"
#include "oscCatalogReference.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscRouteRef : public oscObjectBase
{
public:
oscRouteRef()
{
        OSC_OBJECT_ADD_MEMBER(Route, "oscRoute", 1);
        OSC_OBJECT_ADD_MEMBER(CatalogReference, "oscCatalogReference", 1);
    };
        const char *getScope(){return "/OSCPosition/Route";};
    oscRouteMember Route;
    oscCatalogReferenceMember CatalogReference;

};

typedef oscObjectVariable<oscRouteRef *> oscRouteRefMember;
typedef oscObjectVariableArray<oscRouteRef *> oscRouteRefArrayMember;


}

#endif //OSCROUTEREF_H
