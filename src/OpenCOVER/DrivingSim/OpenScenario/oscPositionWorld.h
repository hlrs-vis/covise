/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_POSITION_WORLD_H
#define OSC_POSITION_WORLD_H

#include "oscExport.h"
#include "oscOrientation.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscPositionWorld: public oscOrientation
{
public:
    oscPositionWorld()
    {
        OSC_ADD_MEMBER(x);
        OSC_ADD_MEMBER(y);
        OSC_ADD_MEMBER(z);
    };

    oscDouble x;
    oscDouble y;
    oscDouble z;
};

typedef oscObjectVariable<oscPositionWorld *> oscPositionWorldMember;

}

#endif //OSC_POSITION_WORLD_H
