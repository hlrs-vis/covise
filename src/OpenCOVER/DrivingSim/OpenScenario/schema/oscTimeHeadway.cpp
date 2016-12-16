/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */

#include "schema/oscTimeHeadway.h"

using namespace OpenScenario;
Enum_ruleType::Enum_ruleType()
{
addEnum("greater_than", oscTimeHeadway::greater_than);
addEnum("less_than", oscTimeHeadway::less_than);
addEnum("equal_to", oscTimeHeadway::equal_to);
}
Enum_ruleType *Enum_ruleType::instance()
{
	if (inst == NULL)
	{
		inst = new Enum_ruleType();
	}
	return inst;
}
Enum_ruleType *Enum_ruleType::inst = NULL;
