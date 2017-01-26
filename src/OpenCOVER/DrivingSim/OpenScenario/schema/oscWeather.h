/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCWEATHER_H
#define OSCWEATHER_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscSun.h"
#include "oscFog.h"
#include "oscPrecipitation.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT Enum_cloudStateType : public oscEnumType
{
public:
static Enum_cloudStateType *instance();
    private:
		Enum_cloudStateType();
	    static Enum_cloudStateType *inst; 
};
class OPENSCENARIOEXPORT oscWeather : public oscObjectBase
{
public:
oscWeather()
{
        OSC_ADD_MEMBER(cloudState, 0);
        OSC_OBJECT_ADD_MEMBER(Sun, "oscSun", 0);
        OSC_OBJECT_ADD_MEMBER(Fog, "oscFog", 0);
        OSC_OBJECT_ADD_MEMBER(Precipitation, "oscPrecipitation", 0);
        cloudState.enumType = Enum_cloudStateType::instance();
    };
    oscEnum cloudState;
    oscSunMember Sun;
    oscFogMember Fog;
    oscPrecipitationMember Precipitation;

    enum Enum_cloudState
    {
sky_off,
free,
cloudy,
overcast,
rainy,

    };

};

typedef oscObjectVariable<oscWeather *> oscWeatherMember;
typedef oscObjectVariableArray<oscWeather *> oscWeatherArrayMember;


}

#endif //OSCWEATHER_H
