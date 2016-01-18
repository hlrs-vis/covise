/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#include <oscPriority.h>


using namespace OpenScenario;


priorityType::priorityType()
{
    addEnum("overwrite", oscPriority::overwrite);
    addEnum("following", oscPriority::following);
    addEnum("skip", oscPriority::skip);
}

priorityType *priorityType::instance()
{
	if(inst == NULL)
	{
		inst = new priorityType();
	}
	return inst;
}

priorityType *priorityType::inst = NULL;
