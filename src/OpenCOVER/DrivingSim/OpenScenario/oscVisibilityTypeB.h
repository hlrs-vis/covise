/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_VISIBILITY_TYPE_B_H
#define OSC_VISIBILITY_TYPE_B_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscVisibilityTypeB: public oscObjectBase
{
public:
    oscVisibilityTypeB()
    {
        OSC_ADD_MEMBER(value);
    };

    oscFloat value;
};

typedef oscObjectVariable<oscVisibilityTypeB *> oscVisibilityTypeBMember;

}

#endif //OSC_VISIBILITY_TYPE_B_H
