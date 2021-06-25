/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCPRIVATEACTION_H
#define OSCPRIVATEACTION_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscLongitudinal.h"
#include "oscLateral.h"
#include "oscVisibility.h"
#include "oscMeeting.h"
#include "oscAutonomous.h"
#include "oscActionController.h"
#include "oscPosition.h"
#include "oscRouting.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscPrivateAction : public oscObjectBase
{
public:
oscPrivateAction()
{
        OSC_OBJECT_ADD_MEMBER(Longitudinal, "oscLongitudinal", 1);
        OSC_OBJECT_ADD_MEMBER(Lateral, "oscLateral", 1);
        OSC_OBJECT_ADD_MEMBER(Visibility, "oscVisibility", 1);
        OSC_OBJECT_ADD_MEMBER(Meeting, "oscMeeting", 1);
        OSC_OBJECT_ADD_MEMBER(Autonomous, "oscAutonomous", 1);
        OSC_OBJECT_ADD_MEMBER(Controller, "oscActionController", 1);
        OSC_OBJECT_ADD_MEMBER(Position, "oscPosition", 1);
        OSC_OBJECT_ADD_MEMBER(Routing, "oscRouting", 1);
    };
        const char *getScope(){return "";};
    oscLongitudinalMember Longitudinal;
    oscLateralMember Lateral;
    oscVisibilityMember Visibility;
    oscMeetingMember Meeting;
    oscAutonomousMember Autonomous;
    oscActionControllerMember Controller;
    oscPositionMember Position;
    oscRoutingMember Routing;

};

typedef oscObjectVariable<oscPrivateAction *> oscPrivateActionMember;
typedef oscObjectVariableArray<oscPrivateAction *> oscPrivateActionArrayMember;


}

#endif //OSCPRIVATEACTION_H
