/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCABSOLUTE_H
#define OSCABSOLUTE_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscAbsolute : public oscObjectBase
{
public:
oscAbsolute()
{
        OSC_ADD_MEMBER(value);
    };
    oscDouble value;

};

typedef oscObjectVariable<oscAbsolute *> oscAbsoluteMember;
typedef oscObjectVariableArray<oscAbsolute *> oscAbsoluteArrayMember;


}

#endif //OSCABSOLUTE_H
