/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_ACCELERATION_H
#define OSC_ACCELERATION_H

#include "oscExport.h"
#include "oscConditionChoiceTypeA.h"
#include "oscObjectVariable.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscAcceleration : public oscConditionChoiceTypeA
{
public:
    oscAcceleration()
    {

    };

};

typedef oscObjectVariable<oscAcceleration *> oscAccelerationMember;

}

#endif //OSC_ACCELERATION_H
