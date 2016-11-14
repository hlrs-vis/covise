/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_STORYBOARD_H
#define OSC_STORYBOARD_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscInit.h"
#include "oscStory.h"
#include "oscEnd.h"

namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscStoryboard: public oscObjectBase
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

#endif //OSC_STORYBOARD_H
