/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCCONTROLPOINT1_H
#define OSCCONTROLPOINT1_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscControlPoint1 : public oscObjectBase
{
public:
oscControlPoint1()
{
        OSC_ADD_MEMBER(status, 0);
    };
        const char *getScope(){return "/OSCTrajectory/Vertex/Shape/Spline";};
    oscString status;

};

typedef oscObjectVariable<oscControlPoint1 *> oscControlPoint1Member;
typedef oscObjectVariableArray<oscControlPoint1 *> oscControlPoint1ArrayMember;


}

#endif //OSCCONTROLPOINT1_H
