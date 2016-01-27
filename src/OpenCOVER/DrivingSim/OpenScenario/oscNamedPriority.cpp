/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#include "oscNamedPriority.h"


using namespace OpenScenario;


maneuverPriorityType::maneuverPriorityType()
{
    addEnum("overwrite", oscNamedPriority::overwrite);
    addEnum("following", oscNamedPriority::following);
    addEnum("cancel", oscNamedPriority::cancel);
}

maneuverPriorityType *maneuverPriorityType::instance()
{
	if(inst == NULL)
	{
		inst = new maneuverPriorityType();
	}
	return inst;
}

maneuverPriorityType *maneuverPriorityType::inst = NULL;
