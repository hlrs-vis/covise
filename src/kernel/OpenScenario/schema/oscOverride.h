/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCOVERRIDE_H
#define OSCOVERRIDE_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscThrottle.h"
#include "oscBrake.h"
#include "oscClutch.h"
#include "oscParkingBrake.h"
#include "oscSteeringWheel.h"
#include "oscGear.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscOverride : public oscObjectBase
{
public:
oscOverride()
{
        OSC_OBJECT_ADD_MEMBER(Throttle, "oscThrottle", 0);
        OSC_OBJECT_ADD_MEMBER(Brake, "oscBrake", 0);
        OSC_OBJECT_ADD_MEMBER(Clutch, "oscClutch", 0);
        OSC_OBJECT_ADD_MEMBER(ParkingBrake, "oscParkingBrake", 0);
        OSC_OBJECT_ADD_MEMBER(SteeringWheel, "oscSteeringWheel", 0);
        OSC_OBJECT_ADD_MEMBER(Gear, "oscGear", 0);
    };
        const char *getScope(){return "/OSCPrivateAction/ActionController";};
    oscThrottleMember Throttle;
    oscBrakeMember Brake;
    oscClutchMember Clutch;
    oscParkingBrakeMember ParkingBrake;
    oscSteeringWheelMember SteeringWheel;
    oscGearMember Gear;

};

typedef oscObjectVariable<oscOverride *> oscOverrideMember;
typedef oscObjectVariableArray<oscOverride *> oscOverrideArrayMember;


}

#endif //OSCOVERRIDE_H
