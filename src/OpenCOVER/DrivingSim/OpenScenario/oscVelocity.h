/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_VELOCITY_H
#define OSC_VELOCITY_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>
#include <oscObjectRef.h>

namespace OpenScenario {


/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscVelocity: public oscObjectBase
{
public:
    oscVelocity()
    {	
		OSC_OBJECT_ADD_MEMBER(objectRef, "oscObjectRef");
    };
	oscObjectRefMember objectRef;
};

typedef oscObjectVariable<oscVelocity *> oscVelocityMember;

}

#endif //OSC_VELOCITY_H
