/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCVEHICLEDISTRIBUTION_H
#define OSCVEHICLEDISTRIBUTION_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscVehicle.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscVehicleDistribution : public oscObjectBase
{
public:
oscVehicleDistribution()
{
        OSC_ADD_MEMBER(category);
        OSC_ADD_MEMBER(percentage);
        OSC_OBJECT_ADD_MEMBER(Vehicle, "oscVehicle");
    };
    oscString category;
    oscDouble percentage;
    oscVehicleArrayMember Vehicle;

};

typedef oscObjectVariable<oscVehicleDistribution *> oscVehicleDistributionMember;
typedef oscObjectVariableArray<oscVehicleDistribution *> oscVehicleDistributionArrayMember;


}

#endif //OSCVEHICLEDISTRIBUTION_H
