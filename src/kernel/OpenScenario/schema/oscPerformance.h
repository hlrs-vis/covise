/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCPERFORMANCE_H
#define OSCPERFORMANCE_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscPerformance : public oscObjectBase
{
public:
oscPerformance()
{
        OSC_ADD_MEMBER(maxSpeed, 0);
        OSC_ADD_MEMBER(maxDeceleration, 0);
        OSC_ADD_MEMBER(mass, 0);
    };
        const char *getScope(){return "/OSCVehicle";};
    oscDouble maxSpeed;
    oscDouble maxDeceleration;
    oscDouble mass;

};

typedef oscObjectVariable<oscPerformance *> oscPerformanceMember;
typedef oscObjectVariableArray<oscPerformance *> oscPerformanceArrayMember;


}

#endif //OSCPERFORMANCE_H
