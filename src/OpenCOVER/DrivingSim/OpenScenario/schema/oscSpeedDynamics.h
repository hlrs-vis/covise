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
        OSC_ADD_MEMBER(shape);
        OSC_ADD_MEMBER(rate);
        shape.enumType = Enum_Dynamics_shapeType::instance();
    };
    oscEnum shape;
    oscDouble rate;

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
