/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */

#include "schema/oscOrientation.h"

using namespace OpenScenario;
Enum_Orientation_typeType::Enum_Orientation_typeType()
{
addEnum("relative", oscOrientation::relative);
addEnum("absolute", oscOrientation::absolute);
}
Enum_Orientation_typeType *Enum_Orientation_typeType::instance()
{
	if (inst == NULL)
	{
		inst = new Enum_Orientation_typeType();
	}
	return inst;
}
Enum_Orientation_typeType *Enum_Orientation_typeType::inst = NULL;
