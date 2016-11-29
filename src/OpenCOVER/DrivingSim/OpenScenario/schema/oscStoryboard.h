/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCSTORYBOARD_H
#define OSCSTORYBOARD_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscInit.h"
#include "schema/oscStory.h"
#include "schema/oscEnd.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscStoryboard : public oscObjectBase
{
public:
    oscStoryboard()
    {
        OSC_OBJECT_ADD_MEMBER(Init, "oscInit");
        OSC_OBJECT_ADD_MEMBER(Story, "oscStory");
        OSC_OBJECT_ADD_MEMBER(End, "oscEnd");
    };
    oscInitMember Init;
    oscStoryMember Story;
    oscEndMember End;

};

typedef oscObjectVariable<oscStoryboard *> oscStoryboardMember;


}

#endif //OSCSTORYBOARD_H
