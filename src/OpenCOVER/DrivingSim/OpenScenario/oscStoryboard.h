/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_STORYBOARD_H
#define OSC_STORYBOARD_H

#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>

#include <oscManeuverList.h>
#include <oscRefActorList.h>
#include <oscUserData.h>


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscStoryboard: public oscObjectBase
{
public:
    oscStoryboard()
    {
       OSC_OBJECT_ADD_MEMBER(maneuverList,"oscManeuverList");
       OSC_OBJECT_ADD_MEMBER(refActor,"oscRefActorList");
       OSC_OBJECT_ADD_MEMBER(userData,"oscUserData");
    };

    oscManeuverListMember maneuverList;
    oscRefActorListMember refActor;
    oscUserDataMember userData;
};

typedef oscObjectVariable<oscStoryboard *> oscStoryboardMember;

}

#endif //OSC_STORYBOARD_H
