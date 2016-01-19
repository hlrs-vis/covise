/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_INIT_STATE_H
#define OSC_INIT_STATE_H

#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>

#include <oscVariables.h>


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscInitState: public oscObjectBase
{
public:
    oscInitState()
    {
        OSC_ADD_MEMBER(trafficLightPhase);
        OSC_ADD_MEMBER(dynamicTrafficSignState);
    };

    oscString trafficLightPhase;
    oscString dynamicTrafficSignState;
};

typedef oscObjectVariable<oscInitState *>oscInitStateMember;

}

#endif //OSC_INIT_STATE_H
