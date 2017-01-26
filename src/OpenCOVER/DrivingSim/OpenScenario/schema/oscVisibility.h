/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCVISIBILITY_H
#define OSCVISIBILITY_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscVisibility : public oscObjectBase
{
public:
oscVisibility()
{
        OSC_ADD_MEMBER(graphics, 0);
        OSC_ADD_MEMBER(traffic, 0);
        OSC_ADD_MEMBER(sensors, 0);
    };
    oscBool graphics;
    oscBool traffic;
    oscBool sensors;

};

typedef oscObjectVariable<oscVisibility *> oscVisibilityMember;
typedef oscObjectVariableArray<oscVisibility *> oscVisibilityArrayMember;


}

#endif //OSCVISIBILITY_H
