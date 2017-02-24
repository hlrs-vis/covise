/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCENVIRONMENT_H
#define OSCENVIRONMENT_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscParameterDeclaration.h"
#include "oscEnvironmentTimeOfDay.h"
#include "oscWeather.h"
#include "oscRoadCondition.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscEnvironment : public oscObjectBase
{
public:
oscEnvironment()
{
        OSC_ADD_MEMBER(name, 0);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(ParameterDeclaration, "oscParameterDeclaration", 0);
        OSC_OBJECT_ADD_MEMBER(TimeOfDay, "oscEnvironmentTimeOfDay", 0);
        OSC_OBJECT_ADD_MEMBER(Weather, "oscWeather", 0);
        OSC_OBJECT_ADD_MEMBER(RoadCondition, "oscRoadCondition", 0);
    };
        const char *getScope(){return "";};
    oscString name;
    oscParameterDeclarationMember ParameterDeclaration;
    oscEnvironmentTimeOfDayMember TimeOfDay;
    oscWeatherMember Weather;
    oscRoadConditionMember RoadCondition;

};

typedef oscObjectVariable<oscEnvironment *> oscEnvironmentMember;
typedef oscObjectVariableArray<oscEnvironment *> oscEnvironmentArrayMember;


}

#endif //OSCENVIRONMENT_H
