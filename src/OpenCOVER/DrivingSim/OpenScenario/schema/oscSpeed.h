/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCSPEED_H
#define OSCSPEED_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscSpeedDynamics.h"
#include "schema/oscTarget.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscSpeed : public oscObjectBase
{
public:
    oscSpeed()
    {
        OSC_OBJECT_ADD_MEMBER(SpeedDynamics, "oscSpeedDynamics");
        OSC_OBJECT_ADD_MEMBER(Target, "oscTarget");
    };
    oscSpeedDynamicsMember SpeedDynamics;
    oscTargetMember Target;

};

typedef oscObjectVariable<oscSpeed *> oscSpeedMember;


}

#endif //OSCSPEED_H
