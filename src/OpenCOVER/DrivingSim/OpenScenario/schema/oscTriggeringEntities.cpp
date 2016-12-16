/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */

#include "schema/oscTriggeringEntities.h"

using namespace OpenScenario;
Enum_TriggeringEntities_ruleType::Enum_TriggeringEntities_ruleType()
{
addEnum("any", oscTriggeringEntities::any);
addEnum("all", oscTriggeringEntities::all);
}
Enum_TriggeringEntities_ruleType *Enum_TriggeringEntities_ruleType::instance()
{
	if (inst == NULL)
	{
		inst = new Enum_TriggeringEntities_ruleType();
	}
	return inst;
}
Enum_TriggeringEntities_ruleType *Enum_TriggeringEntities_ruleType::inst = NULL;
