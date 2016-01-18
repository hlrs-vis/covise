/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#include <oscTrafficSource.h>


using namespace OpenScenario;


distanceType::distanceType()
{
    addEnum("vehicle", oscTrafficSource::vehicle);
    addEnum("pedestrian", oscTrafficSource::pedestrian);
}

distanceType *distanceType::instance()
{
    if(inst == NULL)
    {
        inst = new distanceType();
    }
    return inst;
}

distanceType *distanceType::inst = NULL;
