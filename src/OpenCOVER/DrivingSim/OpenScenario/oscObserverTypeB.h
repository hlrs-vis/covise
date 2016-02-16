/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_OBSERVER_TYPE_B_H
#define OSC_OBSERVER_TYPE_B_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "oscGeneral.h"
#include "oscWaypoints.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscObserverTypeB: public oscObjectBase
{
public:

    oscObserverTypeB()
    {
        OSC_ADD_MEMBER(name);
        OSC_ADD_MEMBER(refId);
        OSC_ADD_MEMBER(type);
        OSC_OBJECT_ADD_MEMBER(general, "oscGeneral");
        OSC_OBJECT_ADD_MEMBER(waypoints, "oscWaypoints");
    };

    oscString name;
    oscInt refId;
    oscString type;
    oscGeneralMember general;
    oscWaypointsMemberArray waypoints;
};

typedef oscObjectVariable<oscObserverTypeB *> oscObserverTypeBMember;

}

#endif /* OSC_OBSERVER_TYPE_B_H */
