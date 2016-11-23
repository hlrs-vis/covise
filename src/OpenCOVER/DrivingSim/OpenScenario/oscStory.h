/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_STORY_H
#define OSC_STORY_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscManeuverGroupsTypeAB.h"
#include "oscRefActorsTypeA.h"
#include "oscUserDataList.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscStory: public oscObjectBase
{
public:
    oscStory()
    {
        OSC_OBJECT_ADD_MEMBER(maneuverGroups, "oscManeuverGroupsTypeAB");
        OSC_OBJECT_ADD_MEMBER(refActors, "oscRefActorsTypeA");     
    };

    oscManeuverGroupsTypeABArrayMember maneuverGroups;
    oscRefActorsTypeAArrayMember refActors;
    
};

typedef oscObjectVariable<oscStory *> oscStoryMember;

}

#endif //OSC_STORY_H
