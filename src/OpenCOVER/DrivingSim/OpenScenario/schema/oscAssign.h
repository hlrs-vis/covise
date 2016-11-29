/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCASSIGN_H
#define OSCASSIGN_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscAssign : public oscObjectBase
{
public:
    oscAssign()
    {
        OSC_ADD_MEMBER(name);
    };
    oscString name;

};

typedef oscObjectVariable<oscAssign *> oscAssignMember;


}

#endif //OSCASSIGN_H
