/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#include <oscWeather.h>


using namespace OpenScenario;


cloudStateType::cloudStateType()
{
    addEnum("sky off", oscWeather::sky_off);
    addEnum("free", oscWeather::free);
    addEnum("cloudy", oscWeather::cloudy);
    addEnum("overcast", oscWeather::overcast);
    addEnum("rainy", oscWeather::rainy);
}

cloudStateType *cloudStateType::instance()
{
    if(inst == NULL)
    {
        inst = new cloudStateType();
    }
    return inst;
}

cloudStateType *cloudStateType::inst = NULL;
