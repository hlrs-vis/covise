/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */

#include "oscPersonDescription.h"

using namespace OpenScenario;
Enum_sexType::Enum_sexType()
{
addEnum("male", oscPersonDescription::male);
addEnum("female", oscPersonDescription::female);
}
Enum_sexType *Enum_sexType::instance()
{
	if (inst == NULL)
	{
		inst = new Enum_sexType();
	}
	return inst;
}
Enum_sexType *Enum_sexType::inst = NULL;
