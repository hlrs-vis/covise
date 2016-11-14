/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_SIGNALS_H
#define OSC_SIGNALS_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "oscController.h"

namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscSignals: public oscObjectBase
{
public:
    oscSignals()
    {
        OSC_OBJECT_ADD_MEMBER(Controller, "oscController");
    };

    oscControllerMember Controller;
};

typedef oscObjectVariable<oscSignals *> oscSignalsMember;

}

#endif //OSC_SIGNALS_H
