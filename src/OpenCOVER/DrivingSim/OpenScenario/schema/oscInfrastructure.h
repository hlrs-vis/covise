/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCINFRASTRUCTURE_H
#define OSCINFRASTRUCTURE_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscSignalSystem.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscInfrastructure : public oscObjectBase
{
public:
    oscInfrastructure()
    {
        OSC_OBJECT_ADD_MEMBER(SignalSystem, "oscSignalSystem");
    };
    oscSignalSystemMember SignalSystem;

};

typedef oscObjectVariable<oscInfrastructure *> oscInfrastructureMember;


}

#endif //OSCINFRASTRUCTURE_H
