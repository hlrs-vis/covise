/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCTRAFFICDEFINITION_H
#define OSCTRAFFICDEFINITION_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscVehicleDistribution.h"
#include "schema/oscDriverDistribution.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscTrafficDefinition : public oscObjectBase
{
public:
    oscTrafficDefinition()
    {
        OSC_ADD_MEMBER(name);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(VehicleDistribution, "oscVehicleDistribution");
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(DriverDistribution, "oscDriverDistribution");
    };
    oscString name;
    oscVehicleDistributionMember VehicleDistribution;
    oscDriverDistributionMember DriverDistribution;

};

typedef oscObjectVariable<oscTrafficDefinition *> oscTrafficDefinitionMember;


}

#endif //OSCTRAFFICDEFINITION_H
