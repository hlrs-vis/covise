/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */

#include "oscEvent.h"

using namespace OpenScenario;
Enum_event_priorityType::Enum_event_priorityType()
{
addEnum("overwrite", oscEvent::overwrite);
addEnum("following", oscEvent::following);
addEnum("skip", oscEvent::skip);
}
Enum_event_priorityType *Enum_event_priorityType::instance()
{
	if (inst == NULL)
	{
		inst = new Enum_event_priorityType();
	}
	return inst;
}
Enum_event_priorityType *Enum_event_priorityType::inst = NULL;
