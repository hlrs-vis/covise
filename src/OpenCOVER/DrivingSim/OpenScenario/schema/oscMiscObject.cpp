/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */

#include "schema/oscMiscObject.h"

using namespace OpenScenario;
Enum_MiscObject_categoryType::Enum_MiscObject_categoryType()
{
addEnum("barrier", oscMiscObject::barrier);
addEnum("guardRail", oscMiscObject::guardRail);
addEnum("other", oscMiscObject::other);
}
Enum_MiscObject_categoryType *Enum_MiscObject_categoryType::instance()
{
	if (inst == NULL)
	{
		inst = new Enum_MiscObject_categoryType();
	}
	return inst;
}
Enum_MiscObject_categoryType *Enum_MiscObject_categoryType::inst = NULL;
