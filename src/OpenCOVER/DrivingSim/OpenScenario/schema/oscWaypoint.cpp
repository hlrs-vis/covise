/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */

#include "oscWaypoint.h"

using namespace OpenScenario;
Enum_Route_strategyType::Enum_Route_strategyType()
{
addEnum("fastest", oscWaypoint::fastest);
addEnum("shortest", oscWaypoint::shortest);
addEnum("leastIntersections", oscWaypoint::leastIntersections);
addEnum("random", oscWaypoint::random);
}
Enum_Route_strategyType *Enum_Route_strategyType::instance()
{
	if (inst == NULL)
	{
		inst = new Enum_Route_strategyType();
	}
	return inst;
}
Enum_Route_strategyType *Enum_Route_strategyType::inst = NULL;
