/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#include <oscRoadConditions.h>

using namespace OpenScenario;

effectType::effectType()
{
    addEnum("dry",oscRoadConditions::dry);
    addEnum("water",oscRoadConditions::water);
    addEnum("snow",oscRoadConditions::snow);
    addEnum("oil",oscRoadConditions::oil);
    addEnum("dirt",oscRoadConditions::dirt);
	addEnum("leaves",oscRoadConditions::leaves);
}

effectType *effectType::instance()
{
	if(inst == NULL) 
		inst = new effectType(); 
	return inst;
}

effectType *effectType::inst=NULL;