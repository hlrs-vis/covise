/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_LANE_OFFSET_DYNAMICS_H
#define OSC_LANE_OFFSET_DYNAMICS_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>

namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscLaneOffsetDynamics: public oscObjectBase
{
public:
    oscLaneOffsetDynamics()
    {
        OSC_ADD_MEMBER(maxLateralAcc);
		OSC_ADD_MEMBER(duration);
    };
    oscDouble maxLateralAcc;
    oscDouble duration;
	
};

typedef oscObjectVariable<oscLaneOffsetDynamics *> oscLaneeOffsetDynamicsMember;

}

#endif //OSC_LANE_OFFSET_DYNAMICS_H
