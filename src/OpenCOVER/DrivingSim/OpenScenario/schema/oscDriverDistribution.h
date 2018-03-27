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
#include "oscDriver.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscDriverDistribution : public oscObjectBase
{
public:
oscDriverDistribution()
{
        OSC_OBJECT_ADD_MEMBER(Driver, "oscDriver", 0);
    };
        const char *getScope(){return "/OSCTrafficDefinition";};
    oscDriverArrayMember Driver;

};

typedef oscObjectVariable<oscDriverDistribution *> oscDriverDistributionMember;
typedef oscObjectVariableArray<oscDriverDistribution *> oscDriverDistributionArrayMember;


}

#endif //OSCDRIVERDISTRIBUTION_H
