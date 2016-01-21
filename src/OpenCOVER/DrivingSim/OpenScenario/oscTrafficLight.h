/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_TRAFFIC_LIGHT_H
#define OSC_TRAFFIC_LIGHT_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscSetState.h"
#include "oscSetController.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscTrafficLight: public oscObjectBase
{
public:
    oscTrafficLight()
    {
        OSC_OBJECT_ADD_MEMBER(setState, "oscSetState");
        OSC_OBJECT_ADD_MEMBER(setController, "oscSetController");
    };

    oscSetStateMember setState;
    oscSetControllerMember setController;
};

typedef oscObjectVariable<oscTrafficLight *> oscTrafficLightMember;

}

#endif //OSC_TRAFFIC_LIGHT_H
