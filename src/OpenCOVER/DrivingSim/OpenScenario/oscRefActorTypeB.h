/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_REF_ACTOR_TYPE_B_H
#define OSC_REF_ACTOR_TYPE_B_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscRefActorTypeB: public oscObjectBase
{
public:
    oscRefActorTypeB()
    {
        OSC_ADD_MEMBER(name);
    };

    oscString name;
};

typedef oscObjectVariable<oscRefActorTypeB *> oscRefActorTypeBMember;

}

#endif //OSC_REF_ACTOR_TYPE_B_H
