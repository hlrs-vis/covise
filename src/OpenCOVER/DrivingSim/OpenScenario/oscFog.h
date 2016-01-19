/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_FOG_H
#define OSC_FOG_H

#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>

#include <oscVariables.h>
#include <oscColor.h>
#include <oscBoundingBox.h>


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscFog: public oscObjectBase
{
public:
    oscFog()
    {
        OSC_OBJECT_ADD_MEMBER(color, "oscColor");
        OSC_OBJECT_ADD_MEMBER(boundingBox, "oscBoundingBox");
        OSC_ADD_MEMBER(visibility);
    };

    oscColorMember color;
    oscBoundingBoxMember boundingBox;
    oscFloat visibility;
};

typedef oscObjectVariable<oscFog *> oscFogMember;

}

#endif //OSC_FOG_H
