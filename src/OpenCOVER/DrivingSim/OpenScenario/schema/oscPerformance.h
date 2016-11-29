/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCPERFORMANCE_H
#define OSCPERFORMANCE_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscPerformance : public oscObjectBase
{
public:
    oscPerformance()
    {
        OSC_ADD_MEMBER(maxSpeed);
        OSC_ADD_MEMBER(maxDeceleration);
        OSC_ADD_MEMBER(mass);
    };
    oscDouble maxSpeed;
    oscDouble maxDeceleration;
    oscDouble mass;

};

typedef oscObjectVariable<oscPerformance *> oscPerformanceMember;


}

#endif //OSCPERFORMANCE_H
