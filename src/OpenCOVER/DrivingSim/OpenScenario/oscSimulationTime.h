/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_SIMULATION_TIME_H
#define OSC_SIMULATION_TIME_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>

namespace OpenScenario {


/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscSimulationTime: public oscObjectBase
{
public:
    oscSimulationTime()
    {
        OSC_ADD_MEMBER(simulationTime);
    };
    oscDouble simulationTime;
};

typedef oscObjectVariable<oscSimulationTime *> oscSimulationTimeMember;

}

#endif //OSC_SIMULATION_TIME_H
