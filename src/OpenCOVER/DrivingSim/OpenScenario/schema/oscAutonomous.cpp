/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */

#include "schema/oscAutonomous.h"

using namespace OpenScenario;
Enum_Controller_domainType::Enum_Controller_domainType()
{
addEnum("longitudinal", oscAutonomous::longitudinal);
addEnum("lateral", oscAutonomous::lateral);
addEnum("both", oscAutonomous::both);
}
Enum_Controller_domainType *Enum_Controller_domainType::instance()
{
	if (inst == NULL)
	{
		inst = new Enum_Controller_domainType();
	}
	return inst;
}
Enum_Controller_domainType *Enum_Controller_domainType::inst = NULL;
