/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */

#include "oscPrecipitation.h"

using namespace OpenScenario;
Enum_Precipitation_typeType::Enum_Precipitation_typeType()
{
addEnum("dry", oscPrecipitation::dry);
addEnum("rain", oscPrecipitation::rain);
addEnum("snow", oscPrecipitation::snow);
}
Enum_Precipitation_typeType *Enum_Precipitation_typeType::instance()
{
	if (inst == NULL)
	{
		inst = new Enum_Precipitation_typeType();
	}
	return inst;
}
Enum_Precipitation_typeType *Enum_Precipitation_typeType::inst = NULL;
