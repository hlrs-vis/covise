/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_COORD_H
#define OSC_COORD_H
#include <oscExport.h>
#include <oscFile.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>
#include <oscPosition.h>
#include <oscOrientation.h>

namespace OpenScenario {

class OpenScenarioBase;

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscCoord: public oscObjectBase
{
public:
    oscCoord()
    {
        OSC_ADD_MEMBER(position);
        OSC_ADD_MEMBER(orientation);
    };
    oscPositionMember position;
    oscOrientationMember orientation;
};

typedef oscObjectVariable<oscCoord *> oscCoordMember;

}

#endif //OSC_COORD_H