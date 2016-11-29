/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCENVIRONMENT_H
#define OSCENVIRONMENT_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscTimeOfDay.h"
#include "schema/oscWeather.h"
#include "schema/oscRoadCondition.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscEnvironment : public oscObjectBase
{
public:
    oscEnvironment()
    {
        OSC_ADD_MEMBER(name);
        OSC_OBJECT_ADD_MEMBER(TimeOfDay, "oscTimeOfDay");
        OSC_OBJECT_ADD_MEMBER(Weather, "oscWeather");
        OSC_OBJECT_ADD_MEMBER(RoadCondition, "oscRoadCondition");
    };
    oscString name;
    oscTimeOfDayMember TimeOfDay;
    oscWeatherMember Weather;
    oscRoadConditionMember RoadCondition;

};

typedef oscObjectVariable<oscEnvironment *> oscEnvironmentMember;


}

#endif //OSCENVIRONMENT_H
