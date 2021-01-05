/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCCOLLISIONBYENTITY_H
#define OSCCOLLISIONBYENTITY_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscCollisionByEntity : public oscObjectBase
{
public:
oscCollisionByEntity()
{
        OSC_ADD_MEMBER(name, 0);
    };
        const char *getScope(){return "/OSCCondition/ByEntity/EntityCondition/Collision";};
    oscString name;

};

typedef oscObjectVariable<oscCollisionByEntity *> oscCollisionByEntityMember;
typedef oscObjectVariableArray<oscCollisionByEntity *> oscCollisionByEntityArrayMember;


}

#endif //OSCCOLLISIONBYENTITY_H
