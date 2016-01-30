/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_VELOCITY_H
#define OSC_VELOCITY_H

#include "oscExport.h"
#include "oscConditionChoiceTypeA.h"
#include "oscObjectVariable.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscVelocity: public oscConditionChoiceTypeA
{
public:
    oscVelocity()
    {

    };

};

typedef oscObjectVariable<oscVelocity *> oscVelocityMember;

}

#endif //OSC_VELOCITY_H
