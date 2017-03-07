/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCRELATIVEROAD_H
#define OSCRELATIVEROAD_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscOrientation.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscRelativeRoad : public oscObjectBase
{
public:
oscRelativeRoad()
{
        OSC_ADD_MEMBER(object, 0);
        OSC_ADD_MEMBER(ds, 0);
        OSC_ADD_MEMBER(dt, 0);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Orientation, "oscOrientation", 0);
    };
        const char *getScope(){return "/OSCPosition";};
    oscString object;
    oscDouble ds;
    oscDouble dt;
    oscOrientationMember Orientation;

};

typedef oscObjectVariable<oscRelativeRoad *> oscRelativeRoadMember;
typedef oscObjectVariableArray<oscRelativeRoad *> oscRelativeRoadArrayMember;


}

#endif //OSCRELATIVEROAD_H
