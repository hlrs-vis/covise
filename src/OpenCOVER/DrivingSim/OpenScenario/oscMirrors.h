/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_MIRRORS_H
#define OSC_MIRRORS_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>
#include <oscCoord.h>

namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscMirrors: public oscObjectBase
{
public:
    oscMirrors()
    {
		OSC_ADD_MEMBER(type);
		OSC_OBJECT_ADD_MEMBER(coord, "oscCoord");
    };
	oscString type;
	oscCoordMember coord;
};

typedef oscObjectVariable<oscMirrors *> oscMirrorsMember;

}

#endif //OSC_MIRRORS_H