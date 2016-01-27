/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_ENTITY_DELETE_H
#define OSC_ENTITY_DELETE_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscEntityDelete: public oscObjectBase
{
public:
    oscEntityDelete()
    {
        OSC_ADD_MEMBER(name);
    };

    oscString name;
};

typedef oscObjectVariable<oscEntityDelete *> oscEntityDeleteMember;

}

#endif //OSC_ENTITY_DELETE_H
