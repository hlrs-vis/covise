/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCSPEEDDYNAMICS_H
#define OSCSPEEDDYNAMICS_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT Enum_Dynamics_shapeType : public oscEnumType
{
public:
static Enum_Dynamics_shapeType *instance();
    private:
		Enum_Dynamics_shapeType();
	    static Enum_Dynamics_shapeType *inst; 
};
class OPENSCENARIOEXPORT oscSpeedDynamics : public oscObjectBase
{
public:
oscSpeedDynamics()
{
        OSC_ADD_MEMBER(shape, 0);
        OSC_ADD_MEMBER_OPTIONAL(rate, 0);
        OSC_ADD_MEMBER_OPTIONAL(time, 0);
        OSC_ADD_MEMBER_OPTIONAL(distance, 0);
        shape.enumType = Enum_Dynamics_shapeType::instance();
    };
        const char *getScope(){return "/OSCPrivateAction/Longitudinal/Speed";};
    oscEnum shape;
    oscDouble rate;
    oscDouble time;
    oscDouble distance;

    enum Enum_Dynamics_shape
    {
linear,
cubic,
sinusoidal,
step,

    };

};

typedef oscObjectVariable<oscSpeedDynamics *> oscSpeedDynamicsMember;
typedef oscObjectVariableArray<oscSpeedDynamics *> oscSpeedDynamicsArrayMember;


}

#endif //OSCSPEEDDYNAMICS_H
