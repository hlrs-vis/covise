/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_DIMENSION_TYPE_B_H
#define OSC_DIMENSION_TYPE_B_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscDimensionTypeB: public oscObjectBase
{
public:
    oscDimensionTypeB()
    {
        OSC_ADD_MEMBER(width);
        OSC_ADD_MEMBER(length);
        OSC_ADD_MEMBER(height);
    };

    oscDouble width;
    oscDouble length;
    oscDouble height;
};

typedef oscObjectVariable<oscDimensionTypeB *> oscDimensionTypeBMember;

}

#endif //OSC_DIMENSION_TYPE_B_H
