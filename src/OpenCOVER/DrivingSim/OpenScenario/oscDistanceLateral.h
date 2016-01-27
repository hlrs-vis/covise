/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_DISTANCE_LATERAL_H
#define OSC_DISTANCE_LATERAL_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscDistanceLateral: public oscObjectBase
{
public:
    oscDistanceLateral()
    {
        OSC_ADD_MEMBER(refObject);
        OSC_ADD_MEMBER(freespace);
        OSC_ADD_MEMBER(distance);
    };

    oscString refObject;
    oscBool freespace;
    oscDouble distance;
};

typedef oscObjectVariable<oscDistanceLateral *> oscDistanceLateralMember;

}

#endif //OSC_DISTANCE_LATERAL_H
