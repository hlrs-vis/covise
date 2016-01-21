/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_MANEUVERS_TYPE_C_H
#define OSC_MANEUVERS_TYPE_C_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectArrayVariable.h"

#include "oscManeuverTypeC.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscManeuversTypeC: public oscObjectBase
{
public:
    oscManeuversTypeC()
    {
        OSC_OBJECT_ADD_MEMBER(maneuver, "oscManeuverTypeC");
    };

    oscManeuverTypeCMember maneuver;
};

typedef oscObjectArrayVariable<oscManeuversTypeC *> oscManeuversTypeCArrayMember;

}

#endif /* OSC_MANEUVERS_TYPE_C_H */
