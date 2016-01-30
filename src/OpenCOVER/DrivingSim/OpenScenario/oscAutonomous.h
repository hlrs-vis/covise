/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_AUTONOMOUS_H
#define OSC_AUTONOMOUS_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscAutonomous: public oscObjectBase
{
public:
    oscAutonomous()
    {
        OSC_ADD_MEMBER(activate);
        OSC_ADD_MEMBER(driver);
    };

    oscBool activate;
    oscString driver;
};

typedef oscObjectVariable<oscAutonomous *> oscAutonomousMember;

}

#endif //OSC_AUTONOMOUS_H
