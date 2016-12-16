/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */

#include "schema/oscAfterTermination.h"

using namespace OpenScenario;
Enum_AfterTermination_ruleType::Enum_AfterTermination_ruleType()
{
addEnum("end", oscAfterTermination::end);
addEnum("cancel", oscAfterTermination::cancel);
addEnum("any", oscAfterTermination::any);
}
Enum_AfterTermination_ruleType *Enum_AfterTermination_ruleType::instance()
{
	if (inst == NULL)
	{
		inst = new Enum_AfterTermination_ruleType();
	}
	return inst;
}
Enum_AfterTermination_ruleType *Enum_AfterTermination_ruleType::inst = NULL;
