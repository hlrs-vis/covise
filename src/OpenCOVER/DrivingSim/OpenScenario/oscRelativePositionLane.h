/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_RELATIVE_POSITION_LANE_H
#define OSC_RELATIVE_POSITION_LANE_H

#include <oscExport.h>
#include <oscOrientation.h>
#include <oscObjectVariable.h>

#include <oscVariables.h>


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscRelativePositionLane: public oscOrientation
{
public:
    oscRelativePositionLane()
    {
        OSC_ADD_MEMBER(refObject);
        OSC_ADD_MEMBER(dLane);
        OSC_ADD_MEMBER(ds);
        OSC_ADD_MEMBER(offset);
        OSC_ADD_MEMBER(relativeOrientation);
    };

    oscString refObject;
    oscInt dLane;
    oscDouble ds;
    oscDouble offset;
    oscBool relativeOrientation;
};

typedef oscObjectVariable<oscRelativePositionLane *> oscRelativePositionLaneMember;

}

#endif //OSC_OSC_RELATIVE_POSITION_LANE_H
