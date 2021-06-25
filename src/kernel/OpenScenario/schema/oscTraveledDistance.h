/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCTRAVELEDDISTANCE_H
#define OSCTRAVELEDDISTANCE_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscTraveledDistance : public oscObjectBase
{
public:
oscTraveledDistance()
{
        OSC_ADD_MEMBER(value, 0);
    };
        const char *getScope(){return "/OSCCondition/ByEntity/EntityCondition";};
    oscDouble value;

};

typedef oscObjectVariable<oscTraveledDistance *> oscTraveledDistanceMember;
typedef oscObjectVariableArray<oscTraveledDistance *> oscTraveledDistanceArrayMember;


}

#endif //OSCTRAVELEDDISTANCE_H
