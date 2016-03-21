/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_ENVIRONMENT_H
#define OSC_ENVIRONMENT_H

#include "oscExport.h"
#include "oscNameRefIdUserData.h"
#include "oscObjectVariable.h"

#include "oscFileHeader.h"
#include "oscTimeOfDay.h"
#include "oscWeather.h"
#include "oscRoadConditionsGroup.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscEnvironment: public oscNameRefIdUserData
{
    
public:
    oscEnvironment()
    {
        OSC_OBJECT_ADD_MEMBER(fileHeader, "oscFileHeader");
        OSC_OBJECT_ADD_MEMBER(timeOfDay, "oscTimeOfDay");
        OSC_OBJECT_ADD_MEMBER(weather, "oscWeather");
        OSC_OBJECT_ADD_MEMBER(roadConditionsGroup, "oscRoadConditionsGroup");
    };

    oscFileHeaderMember fileHeader;
    oscTimeOfDayMember timeOfDay;
    oscWeatherMember weather;
    oscRoadConditionsGroupMember roadConditionsGroup;
};

typedef oscObjectVariable<oscEnvironment *> oscEnvironmentMember;

}

#endif //OSC_ENVIRONMENT_H
