/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_CENTER_H
#define OSC_CENTER_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscCenter: public oscObjectBase
{
public:
    oscCenter()
    {
        OSC_ADD_MEMBER(x);
        OSC_ADD_MEMBER(y);
        OSC_ADD_MEMBER(z);
        OSC_ADD_MEMBER(h);
    };

    oscDouble x;
    oscDouble y;
    oscDouble z;
    oscFloat h;
};

typedef oscObjectVariable<oscCenter *> oscCenterMember;

}

#endif //OSC_CENTER_H
