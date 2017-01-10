/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCCHANGEDYNAMICS_H
#define OSCCHANGEDYNAMICS_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscSpeedDynamics.h"
#include "oscExtent.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscChangeDynamics : public oscObjectBase
{
public:
oscChangeDynamics()
{
        OSC_ADD_MEMBER(shape);
        OSC_OBJECT_ADD_MEMBER(Extent, "oscExtent");
        shape.enumType = Enum_Dynamics_shapeType::instance();
    };
    oscEnum shape;
    oscExtentMember Extent;

};

typedef oscObjectVariable<oscChangeDynamics *> oscChangeDynamicsMember;
typedef oscObjectVariableArray<oscChangeDynamics *> oscChangeDynamicsArrayMember;


}

#endif //OSCCHANGEDYNAMICS_H
