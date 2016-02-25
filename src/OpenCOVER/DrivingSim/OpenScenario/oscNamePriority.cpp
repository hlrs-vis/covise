/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#include "oscNamePriority.h"


using namespace OpenScenario;


priorityManeuverType::priorityManeuverType()
{
    addEnum("overwrite", oscNamePriority::overwrite);
    addEnum("following", oscNamePriority::following);
    addEnum("cancel", oscNamePriority::cancel);
}

priorityManeuverType *priorityManeuverType::instance()
{
	if(inst == NULL)
	{
		inst = new priorityManeuverType();
	}
	return inst;
}

priorityManeuverType *priorityManeuverType::inst = NULL;
