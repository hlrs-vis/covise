/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_GEARBOX_H
#define OSC_GEARBOX_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscGearbox: public oscObjectBase
{
public:
    oscGearbox()
    {
        OSC_ADD_MEMBER(type);
    };

    oscString type;
};

typedef oscObjectVariable<oscGearbox *> oscGearboxMember;

}

#endif //OSC_GEARBOX_H
