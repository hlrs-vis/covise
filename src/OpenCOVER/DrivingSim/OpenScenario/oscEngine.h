/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_ENGINE_H
#define OSC_ENGINE_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscEngine: public oscObjectBase
{
public:
    oscEngine()
    {
        OSC_ADD_MEMBER(type);
        OSC_ADD_MEMBER(power);
        OSC_ADD_MEMBER(maxRpm);
    };

    oscString type;
    oscDouble power;
    oscDouble maxRpm;
};

typedef oscObjectVariable<oscEngine *> oscEngineMember;

}

#endif //OSC_ENGINE_H
