/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */

#include "schema/oscByType.h"

using namespace OpenScenario;
Enum_ObjectTypeType::Enum_ObjectTypeType()
{
addEnum("pedestrian", oscByType::pedestrian);
addEnum("vehicle", oscByType::vehicle);
addEnum("miscellaneous", oscByType::miscellaneous);
}
Enum_ObjectTypeType *Enum_ObjectTypeType::instance()
{
	if (inst == NULL)
	{
		inst = new Enum_ObjectTypeType();
	}
	return inst;
}
Enum_ObjectTypeType *Enum_ObjectTypeType::inst = NULL;
