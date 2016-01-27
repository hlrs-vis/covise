/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_SPEED_DYNAMICS_H
#define OSC_SPEED_DYNAMICS_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscSpeedDynamics: public oscObjectBase
{
public:
    oscSpeedDynamics()
    {
        OSC_ADD_MEMBER(rate);
        OSC_ADD_MEMBER(immediate);
        OSC_ADD_MEMBER(shape);
    };

    oscDouble rate;
    oscBool immediate;
    oscString shape;
};

typedef oscObjectVariable<oscSpeedDynamics *> oscSpeedDynamicsMember;

}

#endif //OSC_SPEED_DYNAMICS_H
