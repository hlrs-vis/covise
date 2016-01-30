/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_VEHICLE_AXLE_H
#define OSC_VEHICLE_AXLE_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscVehicleAxle: public oscObjectBase
{
public:
    oscVehicleAxle()
    {
        OSC_ADD_MEMBER(rollingResistance);
        OSC_ADD_MEMBER(maxSteering);
        OSC_ADD_MEMBER(wheelDiameter);
        OSC_ADD_MEMBER(trackWidth);
        OSC_ADD_MEMBER(positionX);
        OSC_ADD_MEMBER(positionZ);
        OSC_ADD_MEMBER(driven);
    };

    oscDouble rollingResistance;
    oscDouble maxSteering;
    oscDouble wheelDiameter;
    oscDouble trackWidth;
    oscDouble positionX;
    oscDouble positionZ;
    oscBool driven;
};

typedef oscObjectVariable<oscVehicleAxle *> oscVehicleAxleMember;

}

#endif //OSC_VEHICLE_AXLE_H
