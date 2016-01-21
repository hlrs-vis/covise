/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_LANE_OFFSET_H
#define OSC_LANE_OFFSET_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscRelative.h"
#include "oscAbsolute.h"
#include "oscLaneOffsetDynamics.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscLaneOffset: public oscObjectBase
{
public:
    oscLaneOffset()
    {
        OSC_OBJECT_ADD_MEMBER(relative, "oscRelative");
        OSC_OBJECT_ADD_MEMBER(absolute, "oscAbsolute");
        OSC_OBJECT_ADD_MEMBER(dynamics, "oscLaneOffsetDynamics");
    };

    oscRelativeMember relative;
    oscAbsoluteMember absolute;
    oscLaneeOffsetDynamicsMember dynamics;
};

typedef oscObjectVariable<oscLaneOffset *> oscLaneOffsetMember;

}

#endif //OSC_LANE_OFFSET_H
