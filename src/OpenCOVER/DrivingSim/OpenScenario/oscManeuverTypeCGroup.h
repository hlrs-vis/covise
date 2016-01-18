/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_MANEUVER_TYPE_C_GROUP_H
#define OSC_MANEUVER_TYPE_C_GROUP_H

#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectArrayVariable.h>

#include <oscManeuverTypeC.h>


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscManeuverTypeCGroup: public oscObjectBase
{
public:
    oscManeuverTypeCGroup()
    {
        OSC_OBJECT_ADD_MEMBER(maneuver, "oscManeuverTypeC");
    };

    oscManeuverTypeCMember maneuver;
};

typedef oscObjectArrayVariable<oscManeuverTypeCGroup *> oscManeuverTypeCGroupArrayMember;

}

#endif /* OSC_MANEUVER_TYPE_C_GROUP_H */
