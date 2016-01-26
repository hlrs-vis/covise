/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#include "oscPartnerType.h"


using namespace OpenScenario;


partnerTObjTypeType::partnerTObjTypeType()
{
    addEnum("vehicle", oscPartnerType::vehicle);
    addEnum("pedestrian", oscPartnerType::pedestrian);
    addEnum("traffic sign", oscPartnerType::trafficSign);
    addEnum("infrastructure", oscPartnerType::infrastructure);
}

partnerTObjTypeType *partnerTObjTypeType::instance()
{
    if(inst == NULL)
    {
        inst = new partnerTObjTypeType();
    }
    return inst;
}

partnerTObjTypeType *partnerTObjTypeType::inst = NULL;
