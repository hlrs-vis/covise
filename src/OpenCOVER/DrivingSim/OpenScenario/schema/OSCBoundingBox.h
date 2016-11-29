/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCBOUNDINGBOX_H
#define OSCBOUNDINGBOX_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscCenter.h"
#include "schema/oscDimension.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscBoundingBox : public oscObjectBase
{
public:
    oscBoundingBox()
    {
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Center, "oscCenter");
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Dimension, "oscDimension");
    };
    oscCenterMember Center;
    oscDimensionMember Dimension;

};

typedef oscObjectVariable<oscBoundingBox *> oscBoundingBoxMember;


}

#endif //OSCBOUNDINGBOX_H
