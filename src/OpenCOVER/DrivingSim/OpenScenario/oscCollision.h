/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_COLLISION_H
#define OSC_COLLISION_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>
#include <oscPartner.h>

namespace OpenScenario {


/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscCollision: public oscObjectBase
{
public:
    oscCollision()
    {	
		OSC_ADD_MEMBER(object);
		OSC_OBJECT_ADD_MEMBER(partner, "oscPartner");
    };
	oscString object;
	oscPartnerMember partner;
};

typedef oscObjectVariable<oscCollision *> oscCollisionMember;

}

#endif //OSC_COLLISION_H
