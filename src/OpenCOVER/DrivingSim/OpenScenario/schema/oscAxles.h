/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCAXLES_H
#define OSCAXLES_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscAxle.h"
#include "oscAxle.h"
#include "oscAxle.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscAxles : public oscObjectBase
{
public:
oscAxles()
{
        OSC_OBJECT_ADD_MEMBER(Front, "oscAxle", 0);
        OSC_OBJECT_ADD_MEMBER(Rear, "oscAxle", 0);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Additional, "oscAxle", 0);
    };
    oscAxleMember Front;
    oscAxleMember Rear;
    oscAxleArrayMember Additional;

};

typedef oscObjectVariable<oscAxles *> oscAxlesMember;
typedef oscObjectVariableArray<oscAxles *> oscAxlesArrayMember;


}

#endif //OSCAXLES_H
