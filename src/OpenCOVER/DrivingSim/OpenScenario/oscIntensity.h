/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_INTENSITY_H
#define OSC_INTENSITY_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>

namespace OpenScenario {

class OpenScenarioBase;

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscIntensity: public oscObjectBase
{
public:
    oscIntensity()
    {
        OSC_ADD_MEMBER(ambient);
		OSC_ADD_MEMBER(diffuse);
		OSC_ADD_MEMBER(specular);
    };
    oscFloat ambient;
	oscFloat diffuse;
	oscFloat specular;
};

typedef oscObjectVariable<oscIntensity *> oscIntensityMember;

}

#endif //OSC_INTENSITY_H