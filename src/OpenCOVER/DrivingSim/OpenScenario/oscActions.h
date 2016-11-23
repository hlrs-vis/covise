/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_ACTIONS_H
#define OSC_ACTIONS_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariableArray.h"

#include "oscUserDefined.h"
#include "oscGlobal.h"
#include "oscPrivate.h"

namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscActions: public oscObjectBase
{
public:
    oscActions()
    {
        OSC_OBJECT_ADD_MEMBER(Global, "oscGlobal");
		OSC_OBJECT_ADD_MEMBER(UserDefined, "oscUserDefined");
		OSC_OBJECT_ADD_MEMBER(Private, "oscPrivate");
    };

    oscGlobalMember Global;
	oscUserDefinedMember UserDefined;
	oscPrivateMember Private;
};

typedef oscObjectVariable<oscActions *> oscActionsMember;

}

#endif /* OSC_ACTIONS_H */
