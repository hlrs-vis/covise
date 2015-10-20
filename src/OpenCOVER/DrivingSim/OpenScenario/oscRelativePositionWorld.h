/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_RELATIVE_POSITION_WORLD_H
#define OSC_RELATIVE_POSITION_WORLD_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>
#include <oscOrientation.h>

namespace OpenScenario {

class OpenScenarioBase;

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscRelativePisitionWorld: public oscObjectBase
{
public:
    oscRelativePisitionWorld()
    {
        OSC_ADD_MEMBER(refObject);
		OSC_ADD_MEMBER(dx);
		OSC_ADD_MEMBER(dy);
		OSC_ADD_MEMBER(dz);
		OSC_ADD_MEMBER(orientation);		
    };
    oscString refObject;
	oscDouble dx;
	oscDouble dy;
	oscDouble dz;
	oscOrientationMember orientation;
};

typedef oscObjectVariable<oscRelativePisitionWorld *> oscRelativePisitionWorldMember;

}

#endif //OSC_RELATIVE_POSITION_WORLD_H
