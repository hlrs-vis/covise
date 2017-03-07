/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCLIMITED_H
#define OSCLIMITED_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscLimited : public oscObjectBase
{
public:
oscLimited()
{
        OSC_ADD_MEMBER(maxAcceleration, 0);
        OSC_ADD_MEMBER(maxDeceleration, 0);
        OSC_ADD_MEMBER(maxSpeed, 0);
    };
        const char *getScope(){return "/OSCPrivateAction/Longitudinal/DistanceAction/Dynamics";};
    oscDouble maxAcceleration;
    oscDouble maxDeceleration;
    oscDouble maxSpeed;

};

typedef oscObjectVariable<oscLimited *> oscLimitedMember;
typedef oscObjectVariableArray<oscLimited *> oscLimitedArrayMember;


}

#endif //OSCLIMITED_H
