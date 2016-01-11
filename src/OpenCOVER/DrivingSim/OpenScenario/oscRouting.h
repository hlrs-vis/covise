/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_ROUTING_H
#define OSC_ROUTING_H

#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>

#include <oscHeader.h>
#include <oscGeneral.h>
#include <oscWaypoint.h>
#include <oscUserData.h>


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscRouting: public oscObjectBase
{
public:
	
    oscRouting()
    {
		OSC_OBJECT_ADD_MEMBER(header, "oscHeader");
		OSC_OBJECT_ADD_MEMBER(general, "oscGeneral");
		OSC_OBJECT_ADD_MEMBER(waypoint, "oscWaypoint");
		OSC_OBJECT_ADD_MEMBER(userData, "oscUserData");
    };

	oscHeaderMember header;
	oscGeneralMember general;
	oscWaypointMember waypoint;
	oscUserDataMember userData;
};

typedef oscObjectVariable<oscRouting *> oscRoutingMember;

}

#endif //OSC_ROUTING_H
