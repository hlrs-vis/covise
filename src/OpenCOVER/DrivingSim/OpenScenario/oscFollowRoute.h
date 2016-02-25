/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_FOLLOW_ROUTE_H
#define OSC_FOLLOW_ROUTE_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "oscLongitudinal.h"
#include "oscLateral.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscFollowRoute: public oscObjectBase
{
public:
    oscFollowRoute()
    {
        OSC_ADD_MEMBER(routeId);
        OSC_OBJECT_ADD_MEMBER(longitudinal, "oscLongitudinal");
        OSC_OBJECT_ADD_MEMBER(lateral, "oscLateral");
    };

    oscString routeId;
    oscLongitudinalMember longitudinal;
    oscLateralMember lateral;
};

typedef oscObjectVariable<oscFollowRoute *> oscFollowRouteMember;

}

#endif /* OSC_FOLLOW_ROUTE_H */
