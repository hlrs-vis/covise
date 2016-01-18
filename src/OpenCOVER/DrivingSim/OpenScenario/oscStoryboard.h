/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_STORYBOARD_H
#define OSC_STORYBOARD_H

#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>

#include <oscManeuverLists.h>
#include <oscRefActorTypeAList.h>
#include <oscUserDataList.h>


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscStoryboard: public oscObjectBase
{
public:
    oscStoryboard()
    {
        OSC_OBJECT_ADD_MEMBER(maneuverLists, "oscManeuverLists");
        OSC_OBJECT_ADD_MEMBER(refActorList, "oscRefActorTypeAList");
        OSC_OBJECT_ADD_MEMBER(userDataList, "oscUserDataList");
    };

    oscManeuverListsArrayMember maneuverLists;
    oscRefActorTypeAListArrayMember refActorList;
    oscUserDataListArrayMember userDataList;
};

typedef oscObjectVariable<oscStoryboard *> oscStoryboardMember;

}

#endif //OSC_STORYBOARD_H
