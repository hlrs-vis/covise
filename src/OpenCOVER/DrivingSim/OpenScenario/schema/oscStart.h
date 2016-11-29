/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCSTART_H
#define OSCSTART_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscConditionGroup.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscStart : public oscObjectBase
{
public:
    oscStart()
    {
        OSC_OBJECT_ADD_MEMBER(ConditionGroup, "oscConditionGroup");
    };
    oscConditionGroupMember ConditionGroup;

};

typedef oscObjectVariable<oscStart *> oscStartMember;


}

#endif //OSCSTART_H
