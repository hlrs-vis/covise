/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#include <oscEvent.h>

using namespace OpenScenario;

priorityType::priorityType()
{
    addEnum("overwrite",oscEvent::overwrite);
    addEnum("following",oscEvent::following);
    addEnum("skip",oscEvent::skip);
}

priorityType *priorityType::instance()
{
	if(inst == NULL) 
		inst = new priorityType(); 
	return inst;
}

priorityType *priorityType::inst=NULL;