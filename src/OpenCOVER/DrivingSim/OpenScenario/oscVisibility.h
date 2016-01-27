/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_VISIBILITY_H
#define OSC_VISIBILITY_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscVisibility: public oscObjectBase
{
public:
    oscVisibility()
    {
        OSC_ADD_MEMBER(graphics);
        OSC_ADD_MEMBER(traffic);
        OSC_ADD_MEMBER(sensors);
    };

    oscBool graphics;
    oscBool traffic;
    oscBool sensors;
};

typedef oscObjectVariable<oscVisibility *> oscVisibilityMember;

}

#endif //OSC_VISIBILITY_H
