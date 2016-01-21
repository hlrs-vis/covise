/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#include "oscRoadCondition.h"


using namespace OpenScenario;


effectType::effectType()
{
    addEnum("dry", oscRoadCondition::dry);
    addEnum("water", oscRoadCondition::water);
    addEnum("snow", oscRoadCondition::snow);
    addEnum("oil", oscRoadCondition::oil);
    addEnum("dirt", oscRoadCondition::dirt);
    addEnum("leaves", oscRoadCondition::leaves);
}

effectType *effectType::instance()
{
	if(inst == NULL)
	{
		inst = new effectType();
	}
	return inst;
}

effectType *effectType::inst = NULL;
