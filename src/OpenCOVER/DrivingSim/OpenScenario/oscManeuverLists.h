/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_MANEUVER_LISTS_H
#define OSC_MANEUVER_LISTS_H

#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectArrayVariable.h>

#include <oscManeuverList.h>


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscManeuverLists: public oscObjectBase
{
public:
    oscManeuverLists()
    {
        OSC_OBJECT_ADD_MEMBER(maneuverList, "oscManeuverList");
    };

    oscManeuverListMember maneuverList;
};

typedef oscObjectArrayVariable<oscManeuverLists *> oscManeuverListsArrayMember;

}

#endif /* OSC_MANEUVER_LISTS_H */
