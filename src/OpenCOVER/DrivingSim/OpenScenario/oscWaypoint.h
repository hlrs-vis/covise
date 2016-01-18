/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_WAYPOINT_H
#define OSC_WAYPOINT_H

#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>

#include <oscPosition.h>
#include <oscContinuation.h>


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscWaypoint: public oscObjectBase
{
public:
    oscWaypoint()
    {
        OSC_OBJECT_ADD_MEMBER(position, "oscPosition");
        OSC_OBJECT_ADD_MEMBER(continuation, "oscContinuation");
    };

    oscPositionMember position;
    oscContinuationMember continuation;
};

typedef oscObjectVariable<oscWaypoint *> oscWaypointMember;

}

#endif //OSC_WAYPOINT_H
