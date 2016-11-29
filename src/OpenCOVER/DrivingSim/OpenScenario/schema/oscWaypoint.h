/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCWAYPOINT_H
#define OSCWAYPOINT_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscPosition.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT Enum_Route_strategyType : public oscEnumType
{
public:
static Enum_Route_strategyType *instance();
    private:
		Enum_Route_strategyType();
	    static Enum_Route_strategyType *inst; 
};
class OPENSCENARIOEXPORT oscWaypoint : public oscObjectBase
{
public:
    oscWaypoint()
    {
        OSC_ADD_MEMBER(strategy);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Position, "oscPosition");
    };
    oscEnum strategy;
    oscPositionMember Position;

    enum Enum_Route_strategy
    {
fastest,
shortest,
leastIntersections,
random,

    };

};

typedef oscObjectVariable<oscWaypoint *> oscWaypointMember;


}

#endif //OSCWAYPOINT_H
