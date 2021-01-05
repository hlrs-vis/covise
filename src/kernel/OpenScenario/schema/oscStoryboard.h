/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCSTORYBOARD_H
#define OSCSTORYBOARD_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscInit.h"
#include "oscStory.h"
#include "oscEndConditions.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscStoryboard : public oscObjectBase
{
public:
oscStoryboard()
{
        OSC_OBJECT_ADD_MEMBER(Init, "oscInit", 0);
        OSC_OBJECT_ADD_MEMBER(Story, "oscStory", 0);
        OSC_OBJECT_ADD_MEMBER(EndConditions, "oscEndConditions", 0);
    };
        const char *getScope(){return "/OpenSCENARIO";};
    oscInitMember Init;
    oscStoryArrayMember Story;
    oscEndConditionsMember EndConditions;

};

typedef oscObjectVariable<oscStoryboard *> oscStoryboardMember;
typedef oscObjectVariableArray<oscStoryboard *> oscStoryboardArrayMember;


}

#endif //OSCSTORYBOARD_H
