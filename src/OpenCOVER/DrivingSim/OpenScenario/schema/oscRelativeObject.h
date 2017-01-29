/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCRELATIVEOBJECT_H
#define OSCRELATIVEOBJECT_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscOrientation.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscRelativeObject : public oscObjectBase
{
public:
oscRelativeObject()
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

typedef oscObjectVariable<oscRelativeObject *> oscRelativeObjectMember;
typedef oscObjectVariableArray<oscRelativeObject *> oscRelativeObjectArrayMember;


}

#endif //OSCRELATIVEOBJECT_H
