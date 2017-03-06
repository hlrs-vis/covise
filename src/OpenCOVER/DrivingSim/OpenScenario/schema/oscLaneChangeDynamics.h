/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCLANECHANGEDYNAMICS_H
#define OSCLANECHANGEDYNAMICS_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscSpeedDynamics.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscLaneChangeDynamics : public oscObjectBase
{
public:
oscLaneChangeDynamics()
{
        OSC_ADD_MEMBER_OPTIONAL(time, 0);
        OSC_ADD_MEMBER_OPTIONAL(distance, 0);
        OSC_ADD_MEMBER(shape, 0);
        shape.enumType = Enum_Dynamics_shapeType::instance();
    };
        const char *getScope(){return "/OSCPrivateAction/Lateral/LaneChange";};
    oscDouble time;
    oscDouble distance;
    oscEnum shape;

};

typedef oscObjectVariable<oscLaneChangeDynamics *> oscLaneChangeDynamicsMember;
typedef oscObjectVariableArray<oscLaneChangeDynamics *> oscLaneChangeDynamicsArrayMember;


}

#endif //OSCLANECHANGEDYNAMICS_H
