/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCINIT_H
#define OSCINIT_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscActions.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscInit : public oscObjectBase
{
public:
oscInit()
{
        OSC_OBJECT_ADD_MEMBER(Actions, "oscActions", 0);
    };
        const char *getScope(){return "/OpenSCENARIO/Storyboard";};
    oscActionsMember Actions;

};

typedef oscObjectVariable<oscInit *> oscInitMember;
typedef oscObjectVariableArray<oscInit *> oscInitArrayMember;


}

#endif //OSCINIT_H
