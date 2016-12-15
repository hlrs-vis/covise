/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */

#include "schema/oscMeeting.h"

using namespace OpenScenario;
Enum_Meeting_Position_modeType::Enum_Meeting_Position_modeType()
{
addEnum("straight", oscMeeting::straight);
addEnum("route", oscMeeting::route);
}
Enum_Meeting_Position_modeType *Enum_Meeting_Position_modeType::instance()
{
	if (inst == NULL)
	{
		inst = new Enum_Meeting_Position_modeType();
	}
	return inst;
}
Enum_Meeting_Position_modeType *Enum_Meeting_Position_modeType::inst = NULL;
