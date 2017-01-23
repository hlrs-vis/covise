/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */

#include "oscRelativeDistance.h"

using namespace OpenScenario;
Enum_RelativeDistance_typeType::Enum_RelativeDistance_typeType()
{
addEnum("longitudinal", oscRelativeDistance::longitudinal);
addEnum("lateral", oscRelativeDistance::lateral);
addEnum("inertial", oscRelativeDistance::inertial);
}
Enum_RelativeDistance_typeType *Enum_RelativeDistance_typeType::instance()
{
	if (inst == NULL)
	{
		inst = new Enum_RelativeDistance_typeType();
	}
	return inst;
}
Enum_RelativeDistance_typeType *Enum_RelativeDistance_typeType::inst = NULL;
