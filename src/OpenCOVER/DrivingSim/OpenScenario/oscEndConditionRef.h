/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_END_CONDITION_REF_H
#define OSC_END_CONDITION_REF_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscConditionObject.h>
#include <oscVariables.h>

namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscEndConditionRef: public oscConditionObject
{
public:
    oscEndConditionRef()
    {
		OSC_ADD_MEMBER(iid);
		OSC_ADD_MEMBER(groupId);
    };
	oscInt iid;
	oscInt groupId;
};

typedef oscObjectVariable<oscEndConditionRef *> oscEndConditionRefMember;

}

#endif //OSC_END_CONDITION_REF_H

