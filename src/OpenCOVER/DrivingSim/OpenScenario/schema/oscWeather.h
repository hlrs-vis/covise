/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCWEATHER_H
#define OSCWEATHER_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscSun.h"
#include "schema/oscFog.h"
#include "schema/oscPrecipitation.h"

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
        OSC_ADD_MEMBER(cloudState);
        OSC_OBJECT_ADD_MEMBER(Sun, "oscSun");
        OSC_OBJECT_ADD_MEMBER(Fog, "oscFog");
        OSC_OBJECT_ADD_MEMBER(Precipitation, "oscPrecipitation");
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


}

#endif //OSCWEATHER_H
