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
#include "oscEnd.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscStoryboard : public oscObjectBase
{
public:
oscStoryboard()
{
        OSC_OBJECT_ADD_MEMBER(Init, "oscInit");
        OSC_OBJECT_ADD_MEMBER(Story, "oscStory");
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(End, "oscEnd");
    };
    oscInitMember Init;
    oscStoryArrayMember Story;
    oscEndMember End;

};

typedef oscObjectVariable<oscStoryboard *> oscStoryboardMember;
typedef oscObjectVariableArray<oscStoryboard *> oscStoryboardArrayMember;


}

#endif //OSCSTORYBOARD_H
