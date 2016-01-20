/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_MANEUVER_GROUPS_H
#define OSC_MANEUVER_GROUPS_H

#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectArrayVariable.h>

#include <oscManeuverGroupTypeAB.h>


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscManeuverGroupsTypeAB: public oscObjectBase
{
public:
    oscManeuverGroupsTypeAB()
    {
        OSC_OBJECT_ADD_MEMBER(maneuverGroup, "oscManeuverGroupTypeAB");
    };

    oscManeuverGroupTypeABMember maneuverGroup;
};

typedef oscObjectArrayVariable<oscManeuverGroupsTypeAB *> oscManeuverGroupsTypeABArrayMember;

}

#endif /* OSC_MANEUVER_GROUPS_H */
