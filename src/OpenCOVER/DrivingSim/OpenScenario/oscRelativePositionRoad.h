/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_RELATIVE_POSITION_ROAD_H
#define OSC_RELATIVE_POSITION_ROAD_H

#include <oscExport.h>
#include <oscOrientation.h>
#include <oscObjectVariable.h>

#include <oscVariables.h>


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscRelativePositionRoad: public oscOrientation
{
public:
    oscRelativePositionRoad()
    {
        OSC_ADD_MEMBER(refObject);
        OSC_ADD_MEMBER(ds);
        OSC_ADD_MEMBER(dt);
        OSC_ADD_MEMBER(relativeOrientation);
    };

    oscString refObject;
    oscDouble ds;
    oscDouble dt;
    oscBool relativeOrientation;
};

typedef oscObjectVariable<oscRelativePositionRoad *> oscRelativePositionRoadMember;

}

#endif //OSC_RELATIVE_POSITION_ROAD_H
