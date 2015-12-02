/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_STOPPING_DISTANCE_H
#define OSC_STOPPING_DISTANCE_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>
#include <oscObject.h>
#include <oscPosition.h>

namespace OpenScenario {


/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscStoppingDistance: public oscObject
{
public:
    oscStoppingDistance()
    {	
		OSC_ADD_MEMBER(freespace);
		OSC_OBJECT_ADD_MEMBER(position, "oscPosition");
    };
	oscBool freespace;
	oscPositionMember position;
};

typedef oscObjectVariable<oscStoppingDistance *> oscStoppingDistanceMember;

}

#endif //OSC_STOPPING_DISTANCE_H
