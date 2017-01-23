/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */

#include "oscCondition.h"

using namespace OpenScenario;
Enum_Condition_edgeType::Enum_Condition_edgeType()
{
addEnum("rising", oscCondition::rising);
addEnum("falling", oscCondition::falling);
addEnum("any", oscCondition::any);
}
Enum_Condition_edgeType *Enum_Condition_edgeType::instance()
{
	if (inst == NULL)
	{
		inst = new Enum_Condition_edgeType();
	}
	return inst;
}
Enum_Condition_edgeType *Enum_Condition_edgeType::inst = NULL;
