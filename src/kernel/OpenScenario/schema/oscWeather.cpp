/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */

#include "oscWeather.h"

using namespace OpenScenario;
Enum_cloudStateType::Enum_cloudStateType()
{
addEnum("skyOff", oscWeather::skyOff);
addEnum("free", oscWeather::free);
addEnum("cloudy", oscWeather::cloudy);
addEnum("overcast", oscWeather::overcast);
addEnum("rainy", oscWeather::rainy);
}
Enum_cloudStateType *Enum_cloudStateType::instance()
{
	if (inst == NULL)
	{
		inst = new Enum_cloudStateType();
	}
	return inst;
}
Enum_cloudStateType *Enum_cloudStateType::inst = NULL;
