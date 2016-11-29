/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCROUTEPOSITION_H
#define OSCROUTEPOSITION_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscCurrent.h"
#include "schema/oscRoadCoord.h"
#include "schema/oscLaneCoord.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscRoutePosition : public oscObjectBase
{
public:
    oscRoutePosition()
    {
        OSC_OBJECT_ADD_MEMBER(Current, "oscCurrent");
        OSC_OBJECT_ADD_MEMBER(RoadCoord, "oscRoadCoord");
        OSC_OBJECT_ADD_MEMBER(LaneCoord, "oscLaneCoord");
    };
    oscCurrentMember Current;
    oscRoadCoordMember RoadCoord;
    oscLaneCoordMember LaneCoord;

};

typedef oscObjectVariable<oscRoutePosition *> oscRoutePositionMember;


}

#endif //OSCROUTEPOSITION_H
