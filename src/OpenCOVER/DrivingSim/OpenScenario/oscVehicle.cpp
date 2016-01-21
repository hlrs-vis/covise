/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#include "oscVehicle.h"


using namespace OpenScenario;


vehicleClassType::vehicleClassType()
{
    addEnum("car", oscVehicle::car);
    addEnum("van", oscVehicle::van);
    addEnum("truck", oscVehicle::truck);
    addEnum("trailer", oscVehicle::trailer);
    addEnum("bus", oscVehicle::bus);
    addEnum("motorbike", oscVehicle::motorbike);
    addEnum("bicycle", oscVehicle::bicycle);
    addEnum("train", oscVehicle::train);
    addEnum("tram", oscVehicle::tram);
}

vehicleClassType *vehicleClassType::instance()
{
    if(inst == NULL)
    {
        inst = new vehicleClassType();
    }
    return inst;
}

vehicleClassType *vehicleClassType::inst = NULL;
