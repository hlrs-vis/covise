/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_REFACTORS_TYPE_B_H
#define OSC_REFACTORS_TYPE_B_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectArrayVariable.h"

#include "oscRefActorTypeB.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscRefActorsTypeB: public oscObjectBase
{
public:
    oscRefActorsTypeB()
    {
        OSC_OBJECT_ADD_MEMBER(refActor, "oscRefActorTypeB");
    };

    oscRefActorTypeBMember refActor;
};

typedef oscObjectArrayVariable<oscRefActorsTypeB *> oscRefActorsTypeBArrayMember;

}

#endif /* OSC_REFACTORS_TYPE_B_H */
