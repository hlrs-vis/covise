/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_COG_H
#define OSC_COG_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>

namespace OpenScenario {


/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscCog: public oscObjectBase
{
public:
    oscCog()
    {
		OSC_ADD_MEMBER(positionX);
		OSC_ADD_MEMBER(positionY);
		OSC_ADD_MEMBER(positionZ);
		OSC_ADD_MEMBER(mass);
    };
	oscDouble positionX;
    oscDouble positionY;
	oscDouble positionZ;
    oscDouble mass;
};

typedef oscObjectVariable<oscCog *> oscCogMember;

}

#endif //OSC_COG_H