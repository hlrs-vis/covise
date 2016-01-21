/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_PERFORMANCE_H
#define OSC_PERFORMANCE_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "oscAerodynamics.h"
#include "oscEngine.h"
#include "oscCog.h"
#include "oscGearbox.h"


namespace OpenScenario {


/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscPerformance: public oscObjectBase
{
public:
    oscPerformance()
    {
        OSC_ADD_MEMBER(maxSpeed);
        OSC_ADD_MEMBER(maxDeceleration);
        OSC_ADD_MEMBER(overallEfficiency);
        OSC_OBJECT_ADD_MEMBER(aerodynamics, "oscAerodynamics");
        OSC_OBJECT_ADD_MEMBER(engine, "oscEngine");
        OSC_OBJECT_ADD_MEMBER(cog, "oscCog");
        OSC_OBJECT_ADD_MEMBER(gearbox, "oscGearbox");
    };

    oscDouble maxSpeed;
    oscDouble maxDeceleration;
    oscDouble overallEfficiency;
    oscAerodynamicsMember aerodynamics;
    oscEngineMember engine;
    oscCogMember cog;
    oscGearboxMember gearbox;
};

typedef oscObjectVariable<oscPerformance *> oscPerformanceMember;

}

#endif //OSC_PERFORMANCE_H
