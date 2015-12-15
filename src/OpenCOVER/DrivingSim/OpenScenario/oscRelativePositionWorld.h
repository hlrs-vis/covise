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

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscRelativePositionWorld: public oscOrientation
{
public:
    oscRelativePositionWorld()
    {
        OSC_ADD_MEMBER(refObject);
		OSC_ADD_MEMBER(dx);
		OSC_ADD_MEMBER(dy);
		OSC_ADD_MEMBER(dz);
    };
    oscString refObject;
	oscDouble dx;
	oscDouble dy;
	oscDouble dz;
};

typedef oscObjectVariable<oscRelativePositionWorld *> oscRelativePositionWorldMember;

}

#endif //OSC_RELATIVE_POSITION_WORLD_H
