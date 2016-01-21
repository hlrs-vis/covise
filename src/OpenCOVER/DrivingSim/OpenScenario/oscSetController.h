/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_SET_CONTROLLER_H
#define OSC_SET_CONTROLLER_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscSetController: public oscObjectBase
{
public:
    oscSetController()
    {
        OSC_ADD_MEMBER(name);
        OSC_ADD_MEMBER(state);
    };

    oscString name;
    oscString state;
};

typedef oscObjectVariable<oscSetController *> oscSetControllerMember;

}

#endif //OSC_SET_CONTROLLER_H
