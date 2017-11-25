/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCCENTRALOBJECT_H
#define OSCCENTRALOBJECT_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscCentralObject : public oscObjectBase
{
public:
oscCentralObject()
{
        OSC_ADD_MEMBER(name, 0);
    };
        const char *getScope(){return "/OSCGlobalAction/Traffic/Swarm";};
    oscString name;

};

typedef oscObjectVariable<oscCentralObject *> oscCentralObjectMember;
typedef oscObjectVariableArray<oscCentralObject *> oscCentralObjectArrayMember;


}

#endif //OSCCENTRALOBJECT_H
