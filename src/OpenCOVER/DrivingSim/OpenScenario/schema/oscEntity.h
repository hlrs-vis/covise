/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCENTITY_H
#define OSCENTITY_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscAddPosition.h"
#include "schema/oscEmpty.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscEntity : public oscObjectBase
{
public:
    oscEntity()
    {
        OSC_ADD_MEMBER(name);
        OSC_OBJECT_ADD_MEMBER(AddPosition, "oscAddPosition");
        OSC_OBJECT_ADD_MEMBER(Delete, "oscEmpty");
    };
    oscString name;
    oscAddPositionMember AddPosition;
    oscEmptyMember Delete;

};

typedef oscObjectVariable<oscEntity *> oscEntityMember;


}

#endif //OSCENTITY_H
