/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCRELATIVEWORLD_H
#define OSCRELATIVEWORLD_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscOrientation.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscRelativeWorld : public oscObjectBase
{
public:
oscRelativeWorld()
{
        OSC_ADD_MEMBER(object, 0);
        OSC_ADD_MEMBER(dx, 0);
        OSC_ADD_MEMBER(dy, 0);
        OSC_ADD_MEMBER_OPTIONAL(dz, 0);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Orientation, "oscOrientation", 0);
    };
    oscString object;
    oscDouble dx;
    oscDouble dy;
    oscDouble dz;
    oscOrientationMember Orientation;

};

typedef oscObjectVariable<oscRelativeWorld *> oscRelativeWorldMember;
typedef oscObjectVariableArray<oscRelativeWorld *> oscRelativeWorldArrayMember;


}

#endif //OSCRELATIVEWORLD_H
