/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_EYEPOINT_H
#define OSC_EYEPOINT_H

#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>

#include <oscVariables.h>
#include <oscCoord.h>


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscEyepoint : public oscObjectBase
{
public:
    oscEyepoint()
    {
        OSC_ADD_MEMBER(type);
        OSC_OBJECT_ADD_MEMBER(coord, "oscCoord");
    };

    oscString type;
    oscCoordMember coord;
};

typedef oscObjectVariable<oscEyepoint *> oscEyepointMember;

}

#endif //OSC_EYEPOINT_H
