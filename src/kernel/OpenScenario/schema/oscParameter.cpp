/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */

#include "oscParameter.h"

using namespace OpenScenario;
Enum_OSC_Parameter_typeType::Enum_OSC_Parameter_typeType()
{
addEnum("integer", oscParameter::integer);
addEnum("double_t", oscParameter::double_t);
addEnum("string", oscParameter::string);
}
Enum_OSC_Parameter_typeType *Enum_OSC_Parameter_typeType::instance()
{
	if (inst == NULL)
	{
		inst = new Enum_OSC_Parameter_typeType();
	}
	return inst;
}
Enum_OSC_Parameter_typeType *Enum_OSC_Parameter_typeType::inst = NULL;
