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
        OSC_ADD_MEMBER(object);
        OSC_ADD_MEMBER(dx);
        OSC_ADD_MEMBER(dy);
        OSC_ADD_MEMBER_OPTIONAL(dz);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Orientation, "oscOrientation");
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
