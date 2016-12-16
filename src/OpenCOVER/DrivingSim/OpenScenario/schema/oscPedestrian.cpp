/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */

#include "schema/oscPedestrian.h"

using namespace OpenScenario;
Enum_Pedestrian_categoryType::Enum_Pedestrian_categoryType()
{
addEnum("pedestrian", oscPedestrian::pedestrian);
addEnum("wheelchair", oscPedestrian::wheelchair);
addEnum("animal", oscPedestrian::animal);
}
Enum_Pedestrian_categoryType *Enum_Pedestrian_categoryType::instance()
{
	if (inst == NULL)
	{
		inst = new Enum_Pedestrian_categoryType();
	}
	return inst;
}
Enum_Pedestrian_categoryType *Enum_Pedestrian_categoryType::inst = NULL;
