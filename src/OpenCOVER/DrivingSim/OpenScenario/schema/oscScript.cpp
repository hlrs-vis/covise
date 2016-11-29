/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */

#include "schema/oscScript.h"

using namespace OpenScenario;
Enum_Script_executionType::Enum_Script_executionType()
{
addEnum("single", oscScript::single);
addEnum("continuous", oscScript::continuous);
}
Enum_Script_executionType *Enum_Script_executionType::instance()
{
	if (inst == NULL)
	{
		inst = new Enum_Script_executionType();
	}
	return inst;
}
Enum_Script_executionType *Enum_Script_executionType::inst = NULL;
