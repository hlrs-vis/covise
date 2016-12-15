/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCCONTROLLER_H
#define OSCCONTROLLER_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"
#include "oscObjectVariableArray.h"

#include "oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscController : public oscObjectBase
{
public:
oscController()
{
        OSC_ADD_MEMBER(name);
        OSC_ADD_MEMBER(state);
    };
    oscString name;
    oscString state;

};

typedef oscObjectVariable<oscController *> oscControllerMember;
typedef oscObjectVariableArray<oscController *> oscControllerArrayMember;


}

#endif //OSCCONTROLLER_H
