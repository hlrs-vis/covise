/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */

#include "schema/oscByCondition.h"

using namespace OpenScenario;
Enum_ByCondition_actorType::Enum_ByCondition_actorType()
{
addEnum("triggeringEntity", oscByCondition::triggeringEntity);
addEnum("anyEntity", oscByCondition::anyEntity);
}
Enum_ByCondition_actorType *Enum_ByCondition_actorType::instance()
{
	if (inst == NULL)
	{
		inst = new Enum_ByCondition_actorType();
	}
	return inst;
}
Enum_ByCondition_actorType *Enum_ByCondition_actorType::inst = NULL;
