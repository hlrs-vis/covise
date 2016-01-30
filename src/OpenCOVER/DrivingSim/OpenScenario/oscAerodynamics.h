/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_AERODYNAMICS_H
#define OSC_AERODYNAMICS_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscAerodynamics: public oscObjectBase
{
public:
    oscAerodynamics()
    {
        OSC_ADD_MEMBER(airDragCoefficient);
        OSC_ADD_MEMBER(frontSurfaceEffective);
    };

    oscDouble airDragCoefficient;
    oscDouble frontSurfaceEffective;
};

typedef oscObjectVariable<oscAerodynamics *> oscAerodynamicsMember;

}

#endif //OSC_AERODYNAMICS_H
