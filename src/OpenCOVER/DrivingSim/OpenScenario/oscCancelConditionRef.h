/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_CANCEL_CONDITION_REF_H
#define OSC_CANCEL_CONDITION_REF_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscConditionObject.h>
#include <oscVariables.h>

namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscCancelConditionRef: public oscConditionObject
{
public:
    oscCancelConditionRef()
    {
		OSC_ADD_MEMBER(iid);
		OSC_ADD_MEMBER(groupId);
    };
	oscInt iid;
	oscInt groupId;
};

typedef oscObjectVariable<oscCancelConditionRef *> oscCancelConditionRefMember;

}

#endif //OSC_CANCEL_CONDITION_REF_H

