/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */

#include "oscByType.h"

using namespace OpenScenario;
oscObjectTypeType::oscObjectTypeType()
{
addEnum("pedestrian", oscByType::pedestrian);
addEnum("vehicle", oscByType::vehicle);
addEnum("miscellaneous", oscByType::miscellaneous);
}
oscObjectTypeType *oscObjectTypeType::instance()
{
	if (inst == NULL)
	{
		inst = new oscObjectTypeType();
	}
	return inst;
}
oscObjectTypeType *oscObjectTypeType::inst = NULL;
