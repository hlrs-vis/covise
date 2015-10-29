/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#include <oscPedestrian.h>

using namespace OpenScenario;

pedestrianClassType::pedestrianClassType()
{
    addEnum("pedestrian",oscPedestrian::pedestrian);
    addEnum("wheelchair",oscPedestrian::wheelchair);
    addEnum("animal",oscPedestrian::animal);
}

pedestrianClassType *pedestrianClassType::instance()
{
    if(inst == NULL) 
        inst = new pedestrianClassType(); 
    return inst;
}

pedestrianClassType *pedestrianClassType::inst=NULL;