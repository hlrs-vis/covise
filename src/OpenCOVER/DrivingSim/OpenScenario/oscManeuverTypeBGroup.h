/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_MANEUVER_TYPE_B_GROUP_H
#define OSC_MANEUVER_TYPE_B_GROUP_H

#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectArrayVariable.h>

#include <oscManeuverTypeB.h>


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscManeuverTypeBGroup: public oscObjectBase
{
public:
    oscManeuverTypeBGroup()
    {
        OSC_OBJECT_ADD_MEMBER(maneuver, "oscManeuverTypeB");
    };

    oscManeuverTypeBMember maneuver;
};

typedef oscObjectArrayVariable<oscManeuverTypeBGroup *> oscManeuverTypeBGroupArrayMember;

}

#endif /* OSC_MANEUVER_TYPE_B_GROUP_H */
