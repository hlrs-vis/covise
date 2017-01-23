/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCBYNAMEDENTITY_H
#define OSCBYNAMEDENTITY_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscByNamedEntity : public oscObjectBase
{
public:
oscByNamedEntity()
{
        OSC_ADD_MEMBER(name);
    };
    oscString name;

};

typedef oscObjectVariable<oscByNamedEntity *> oscByNamedEntityMember;
typedef oscObjectVariableArray<oscByNamedEntity *> oscByNamedEntityArrayMember;


}

#endif //OSCBYNAMEDENTITY_H
