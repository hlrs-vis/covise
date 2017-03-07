/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCDRIVERDISTRIBUTION_H
#define OSCDRIVERDISTRIBUTION_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscVehicle.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscDriverDistribution : public oscObjectBase
{
public:
oscDriverDistribution()
{
        OSC_ADD_MEMBER(name, 0);
        OSC_ADD_MEMBER(percentage, 0);
        OSC_OBJECT_ADD_MEMBER(Vehicle, "oscVehicle", 0);
    };
        const char *getScope(){return "/OSCTrafficDefinition";};
    oscString name;
    oscDouble percentage;
    oscVehicleArrayMember Vehicle;

};

typedef oscObjectVariable<oscDriverDistribution *> oscDriverDistributionMember;
typedef oscObjectVariableArray<oscDriverDistribution *> oscDriverDistributionArrayMember;


}

#endif //OSCDRIVERDISTRIBUTION_H
