/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCENTITY_H
#define OSCENTITY_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscAddPosition.h"
#include "oscEmpty.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscEntity : public oscObjectBase
{
public:
oscEntity()
{
        OSC_ADD_MEMBER(name, 0);
        OSC_OBJECT_ADD_MEMBER(AddPosition, "oscAddPosition", 1);
        OSC_OBJECT_ADD_MEMBER(Delete, "oscEmpty", 1);
    };
    oscString name;
    oscAddPositionMember AddPosition;
    oscEmptyMember Delete;

};

typedef oscObjectVariable<oscEntity *> oscEntityMember;
typedef oscObjectVariableArray<oscEntity *> oscEntityArrayMember;


}

#endif //OSCENTITY_H
