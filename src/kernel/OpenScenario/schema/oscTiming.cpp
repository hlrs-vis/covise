/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */

#include "oscTiming.h"

using namespace OpenScenario;
Enum_domain_absolute_relativeType::Enum_domain_absolute_relativeType()
{
addEnum("absolute", oscTiming::absolute);
addEnum("relative", oscTiming::relative);
}
Enum_domain_absolute_relativeType *Enum_domain_absolute_relativeType::instance()
{
	if (inst == NULL)
	{
		inst = new Enum_domain_absolute_relativeType();
	}
	return inst;
}
Enum_domain_absolute_relativeType *Enum_domain_absolute_relativeType::inst = NULL;
