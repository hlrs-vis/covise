/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCPOSITIONROUTE_H
#define OSCPOSITIONROUTE_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscOrientation.h"
#include "oscRoutePosition.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscPositionRoute : public oscObjectBase
{
public:
oscPositionRoute()
{
        OSC_ADD_MEMBER(route, 0);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Orientation, "oscOrientation", 0);
        OSC_OBJECT_ADD_MEMBER(Position, "oscRoutePosition", 0);
    };
        const char *getScope(){return "/OSCPosition";};
    oscString route;
    oscOrientationMember Orientation;
    oscRoutePositionMember Position;

};

typedef oscObjectVariable<oscPositionRoute *> oscPositionRouteMember;
typedef oscObjectVariableArray<oscPositionRoute *> oscPositionRouteArrayMember;


}

#endif //OSCPOSITIONROUTE_H
