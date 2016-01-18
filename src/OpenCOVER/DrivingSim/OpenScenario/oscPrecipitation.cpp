/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#include <oscPrecipitation.h>


using namespace OpenScenario;


precipitationType::precipitationType()
{
    addEnum("dry", oscPrecipitation::dry);
    addEnum("rain", oscPrecipitation::rain);
    addEnum("snow", oscPrecipitation::snow);
}

precipitationType *precipitationType::instance()
{
	if(inst == NULL)
	{
		inst = new precipitationType();
	}
	return inst;
}

precipitationType *precipitationType::inst = NULL;
