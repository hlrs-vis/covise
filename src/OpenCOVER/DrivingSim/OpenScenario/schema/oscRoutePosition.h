/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCROUTEPOSITION_H
#define OSCROUTEPOSITION_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscRouteRef.h"
#include "oscOrientation.h"
#include "oscPosition.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscRoutePosition : public oscObjectBase
{
public:
oscRoutePosition()
{
        OSC_OBJECT_ADD_MEMBER(RouteRef, "oscRouteRef", 0);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Orientation, "oscOrientation", 0);
        OSC_OBJECT_ADD_MEMBER(Position, "oscPosition", 0);
    };
        const char *getScope(){return "/OSCPosition";};
    oscRouteRefMember RouteRef;
    oscOrientationMember Orientation;
    oscPositionMember Position;

};

typedef oscObjectVariable<oscRoutePosition *> oscRoutePositionMember;
typedef oscObjectVariableArray<oscRoutePosition *> oscRoutePositionArrayMember;


}

#endif //OSCROUTEPOSITION_H
