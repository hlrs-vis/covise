/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_MANEUVERS_TYPE_B_H
#define OSC_MANEUVERS_TYPE_B_H

#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectArrayVariable.h>

#include <oscManeuverTypeB.h>


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscManeuversTypeB: public oscObjectBase
{
public:
    oscManeuversTypeB()
    {
        OSC_OBJECT_ADD_MEMBER(maneuver, "oscManeuverTypeB");
    };

    oscManeuverTypeBMember maneuver;
};

typedef oscObjectArrayVariable<oscManeuversTypeB *> oscManeuversTypeBArrayMember;

}

#endif /* OSC_MANEUVERS_TYPE_B_H */
