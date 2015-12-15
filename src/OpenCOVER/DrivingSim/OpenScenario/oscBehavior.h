/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_BEHAVIOR_H
#define OSC_BEHAVIOR_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>

namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscBehavior: public oscObjectBase
{
public:
    oscBehavior()
    {
        OSC_ADD_MEMBER(politeness);
		OSC_ADD_MEMBER(alertness);
    };
    oscDouble politeness;
	oscDouble alertness;
};

typedef oscObjectVariable<oscBehavior *> oscBehaviorMember;

}

#endif //OSC_BEHAVIOR_H
