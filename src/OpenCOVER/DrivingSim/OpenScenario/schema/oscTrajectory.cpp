/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */

#include "oscTrajectory.h"

using namespace OpenScenario;
Enum_domain_time_distanceType::Enum_domain_time_distanceType()
{
addEnum("time", oscTrajectory::time);
addEnum("distance", oscTrajectory::distance);
}
Enum_domain_time_distanceType *Enum_domain_time_distanceType::instance()
{
	if (inst == NULL)
	{
		inst = new Enum_domain_time_distanceType();
	}
	return inst;
}
Enum_domain_time_distanceType *Enum_domain_time_distanceType::inst = NULL;
