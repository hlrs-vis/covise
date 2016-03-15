/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_CURRENT_POSITION_H
#define OSC_CURRENT_POSITION_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscCurrentPosition: public oscObjectBase
{
public:
    oscCurrentPosition()
    {
        OSC_ADD_MEMBER(current);
    };

    oscBool current;
};

typedef oscObjectVariable<oscCurrentPosition *> oscCurrentPositionMember;

}

#endif /* OSC_CURRENT_POSITION_H */
