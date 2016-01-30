/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_BOUNDING_BOX_H
#define OSC_BOUNDING_BOX_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscCenter.h"
#include "oscDimensionTypeB.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscBoundingBox: public oscObjectBase
{
public:
    oscBoundingBox()
    {
        OSC_OBJECT_ADD_MEMBER(center, "oscCenter");
        OSC_OBJECT_ADD_MEMBER(dimension, "oscDimensionTypeB");
    };

    oscCenterMember center;
    oscDimensionTypeBMember dimension;
};

typedef oscObjectVariable<oscBoundingBox *> oscBoundingBoxMember;

}

#endif //OSC_BOUNDING_BOX_H
