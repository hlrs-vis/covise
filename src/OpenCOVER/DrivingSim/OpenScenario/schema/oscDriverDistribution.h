/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCDRIVERDISTRIBUTION_H
#define OSCDRIVERDISTRIBUTION_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscVehicle.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscDriverDistribution : public oscObjectBase
{
public:
    oscDriverDistribution()
    {
        OSC_ADD_MEMBER(name);
        OSC_ADD_MEMBER(percentage);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Vehicle, "oscVehicle");
    };
    oscString name;
    oscDouble percentage;
    oscVehicleMember Vehicle;

};

typedef oscObjectVariable<oscDriverDistribution *> oscDriverDistributionMember;


}

#endif //OSCDRIVERDISTRIBUTION_H
