/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCDRIVERACTION_H
#define OSCDRIVERACTION_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscAssign.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscDriverAction : public oscObjectBase
{
public:
    oscDriverAction()
    {
        OSC_OBJECT_ADD_MEMBER(Assign, "oscAssign");
    };
    oscAssignMember Assign;

};

typedef oscObjectVariable<oscDriverAction *> oscDriverActionMember;


}

#endif //OSCDRIVERACTION_H
