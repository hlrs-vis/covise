/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */

#include "schema/oscAtStart.h"

using namespace OpenScenario;
Enum_Story_Element_typeType::Enum_Story_Element_typeType()
{
addEnum("act", oscAtStart::act);
addEnum("scene", oscAtStart::scene);
addEnum("maneuver", oscAtStart::maneuver);
addEnum("event", oscAtStart::event);
addEnum("action", oscAtStart::action);
}
Enum_Story_Element_typeType *Enum_Story_Element_typeType::instance()
{
	if (inst == NULL)
	{
		inst = new Enum_Story_Element_typeType();
	}
	return inst;
}
Enum_Story_Element_typeType *Enum_Story_Element_typeType::inst = NULL;
