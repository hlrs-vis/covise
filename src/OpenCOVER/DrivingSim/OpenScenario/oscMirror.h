/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_MIRROR_H
#define OSC_MIRROR_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>
#include <oscCoord.h>

namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscMirror: public oscObjectBase
{
public:
    oscMirror()
    {
		OSC_ADD_MEMBER(type);
		OSC_OBJECT_ADD_MEMBER(coord, "oscCoord");
    };
	oscString type;
	oscCoordMember coord;
};

typedef oscObjectVariable<oscMirror *> oscMirrorMember;

}

#endif //OSC_MIRROR_H