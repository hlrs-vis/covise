/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_LIGHT_H
#define OSC_LIGHT_H

#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>

#include <oscColor.h>
#include <oscIntensity.h>


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscLight: public oscObjectBase
{
public:
    oscLight()
    {
        OSC_OBJECT_ADD_MEMBER(color, "oscColor");
        OSC_OBJECT_ADD_MEMBER(intensity, "oscIntensity");
    };

    oscColorMember color;
    oscIntensityMember intensity;
};

typedef oscObjectVariable<oscLight *> oscLightMember;

}

#endif //OSC_LIGHT_H
