/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCPOSITIONROUTE_H
#define OSCPOSITIONROUTE_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscOrientation.h"
#include "schema/oscRoutePosition.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscPositionRoute : public oscObjectBase
{
public:
    oscPositionRoute()
    {
        OSC_ADD_MEMBER(route);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Orientation, "oscOrientation");
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(RoutePosition, "oscRoutePosition");
    };
    oscString route;
    oscOrientationMember Orientation;
    oscRoutePositionMember RoutePosition;

};

typedef oscObjectVariable<oscPositionRoute *> oscPositionRouteMember;


}

#endif //OSCPOSITIONROUTE_H
