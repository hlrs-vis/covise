/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_COLOR_H
#define OSC_COLOR_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>

namespace OpenScenario {

class OpenScenarioBase;

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscColor: public oscObjectBase
{
public:
    oscColor()
    {
        OSC_ADD_MEMBER(r);
		OSC_ADD_MEMBER(g);
		OSC_ADD_MEMBER(b);
    };
    oscFloat r;
	oscFloat g;
	oscFloat b;
};

typedef oscObjectVariable<oscColor *> oscColorMember;

}

#endif //OSC_COLOR_H