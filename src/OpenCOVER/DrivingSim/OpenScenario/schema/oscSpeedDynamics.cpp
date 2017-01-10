/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */

#include "schema/oscSpeedDynamics.h"

using namespace OpenScenario;
Enum_Dynamics_shapeType::Enum_Dynamics_shapeType()
{
addEnum("linear", oscSpeedDynamics::linear);
addEnum("cubic", oscSpeedDynamics::cubic);
addEnum("sinusoidal", oscSpeedDynamics::sinusoidal);
addEnum("step", oscSpeedDynamics::step);
}
Enum_Dynamics_shapeType *Enum_Dynamics_shapeType::instance()
{
	if (inst == NULL)
	{
		inst = new Enum_Dynamics_shapeType();
	}
	return inst;
}
Enum_Dynamics_shapeType *Enum_Dynamics_shapeType::inst = NULL;
