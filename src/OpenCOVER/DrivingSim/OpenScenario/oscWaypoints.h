/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_WAYPOINTS_H
#define OSC_WAYPOINTS_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariableArray.h"

#include "oscWaypoint.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscWaypoints: public oscObjectBase
{
public:
    oscWaypoints()
    {
        OSC_OBJECT_ADD_MEMBER(waypoint, "oscWaypoint");
    };

    oscWaypointMember waypoint;
};

typedef oscObjectVariableArray<oscWaypoints *> oscWaypointsMemberArray;

}

#endif /* OSC_WAYPOINTS_H */
