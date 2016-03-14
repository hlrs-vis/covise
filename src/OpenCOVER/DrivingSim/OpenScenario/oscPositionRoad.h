/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_POSITION_ROAD_H
#define OSC_POSITION_ROAD_H

#include "oscExport.h"
#include "oscOrientationOptional.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscPositionRoad: public oscOrientationOptional
{
public:
    oscPositionRoad()
    {
        OSC_ADD_MEMBER(roadId);
        OSC_ADD_MEMBER(s);
        OSC_ADD_MEMBER(t);
        OSC_ADD_MEMBER(relativeOrientation);
    };

    oscString roadId;
    oscDouble s;
    oscDouble t;
    oscBool relativeOrientation;
};

typedef oscObjectVariable<oscPositionRoad *> oscPositionRoadMember;

}

#endif //OSC_POSITION_ROAD_H
