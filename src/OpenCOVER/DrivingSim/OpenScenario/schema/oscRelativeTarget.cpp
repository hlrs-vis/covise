/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */

#include "oscRelativeTarget.h"

using namespace OpenScenario;
Enum_Speed_Target_valueTypeType::Enum_Speed_Target_valueTypeType()
{
addEnum("delta", oscRelativeTarget::delta);
addEnum("factor", oscRelativeTarget::factor);
}
Enum_Speed_Target_valueTypeType *Enum_Speed_Target_valueTypeType::instance()
{
	if (inst == NULL)
	{
		inst = new Enum_Speed_Target_valueTypeType();
	}
	return inst;
}
Enum_Speed_Target_valueTypeType *Enum_Speed_Target_valueTypeType::inst = NULL;
