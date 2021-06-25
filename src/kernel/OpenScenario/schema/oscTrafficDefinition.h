/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCTRAFFICDEFINITION_H
#define OSCTRAFFICDEFINITION_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscVehicleDistribution.h"
#include "oscDriverDistribution.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscTrafficDefinition : public oscObjectBase
{
public:
oscTrafficDefinition()
{
        OSC_ADD_MEMBER(name, 0);
        OSC_OBJECT_ADD_MEMBER(VehicleDistribution, "oscVehicleDistribution", 0);
        OSC_OBJECT_ADD_MEMBER(DriverDistribution, "oscDriverDistribution", 0);
    };
        const char *getScope(){return "";};
    oscString name;
    oscVehicleDistributionMember VehicleDistribution;
    oscDriverDistributionMember DriverDistribution;

};

typedef oscObjectVariable<oscTrafficDefinition *> oscTrafficDefinitionMember;
typedef oscObjectVariableArray<oscTrafficDefinition *> oscTrafficDefinitionArrayMember;


}

#endif //OSCTRAFFICDEFINITION_H
