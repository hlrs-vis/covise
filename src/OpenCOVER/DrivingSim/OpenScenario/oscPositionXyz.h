/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_POSITION_XYZ_H
#define OSC_POSITION_XYZ_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscPositionXyz: public oscObjectBase
{
public:
    oscPositionXyz()
    {
        OSC_ADD_MEMBER(x);
        OSC_ADD_MEMBER(y);
        OSC_ADD_MEMBER(z);
    };

    oscDouble x;
    oscDouble y;
    oscDouble z;
};

typedef oscObjectVariable<oscPositionXyz *> oscPositionXyzMember;

}

#endif //OSC_POSITION_XYZ_H
