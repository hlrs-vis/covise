/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCFOLLOWROUTE_H
#define OSCFOLLOWROUTE_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscFollowRoute : public oscObjectBase
{
public:
oscFollowRoute()
{
        OSC_ADD_MEMBER(name);
    };
    oscString name;

};

typedef oscObjectVariable<oscFollowRoute *> oscFollowRouteMember;
typedef oscObjectVariableArray<oscFollowRoute *> oscFollowRouteArrayMember;


}

#endif //OSCFOLLOWROUTE_H
