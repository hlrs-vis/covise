/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_RELATIVE_TYPE_A_H
#define OSC_RELATIVE_TYPE_A_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscRelativeTypeA: public oscObjectBase
{
public:
    oscRelativeTypeA()
    {
        OSC_ADD_MEMBER(refObject);
        OSC_ADD_MEMBER(delta);
    };

    oscString refObject;
    oscInt delta;
};

typedef oscObjectVariable<oscRelativeTypeA *> oscRelativeTypeAMember;

}

#endif //OSC_RELATIVE_TYPE_A_H
