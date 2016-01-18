/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#include <oscPartner.h>


using namespace OpenScenario;


objectTypeType::objectTypeType()
{
    addEnum("vehicle", oscPartner::vehicle);
    addEnum("pedestrian", oscPartner::pedestrian);
    addEnum("trafficSign", oscPartner::trafficSign);
    addEnum("infrastructure", oscPartner::infrastructure);
}

objectTypeType *objectTypeType::instance()
{
    if(inst == NULL)
    {
        inst = new objectTypeType();
    }
    return inst;
}

objectTypeType *objectTypeType::inst = NULL;
