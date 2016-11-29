/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCACTIONS_H
#define OSCACTIONS_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscGlobalAction.h"
#include "schema/oscGlobalAction.h"
#include "schema/oscPrivate.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscActions : public oscObjectBase
{
public:
    oscActions()
    {
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Global, "oscGlobalAction");
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(UserDefined, "oscGlobalAction");
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Private, "oscPrivate");
    };
    oscGlobalActionMember Global;
    oscGlobalActionMember UserDefined;
    oscPrivateMember Private;

};

typedef oscObjectVariable<oscActions *> oscActionsMember;


}

#endif //OSCACTIONS_H
