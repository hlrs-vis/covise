/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_BODY_H
#define OSC_BODY_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>

namespace OpenScenario {

class OpenScenarioBase;

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscBody: public oscObjectBase
{
public:
    oscBody()
    {
        OSC_ADD_MEMBER(weight);
		OSC_ADD_MEMBER(height);
		OSC_ADD_MEMBER(eyeDistance);
    };
    oscDouble weight;
	oscDouble height;
	oscDouble eyeDistance;
};

typedef oscObjectVariable<oscBody *> oscBodyMember;

}

#endif //OSC_BODY_H