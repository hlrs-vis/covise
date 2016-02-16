/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_AXLES_H
#define OSC_AXLES_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVehicleAxle.h"
#include "oscAdditionalAxles.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscAxles: public oscObjectBase
{
public:
    oscAxles()
    {
        OSC_OBJECT_ADD_MEMBER(front, "oscVehicleAxle");
        OSC_OBJECT_ADD_MEMBER(rear, "oscVehicleAxle");
        OSC_OBJECT_ADD_MEMBER(additionalAxles, "oscAdditionalAxles");
    };

    oscVehicleAxleMember front;
    oscVehicleAxleMember rear;
    oscAdditionalAxlesMemberArray additionalAxles;
};

typedef oscObjectVariable<oscAxles *> oscAxlesMember;

}

#endif //OSC_AXLES_H
