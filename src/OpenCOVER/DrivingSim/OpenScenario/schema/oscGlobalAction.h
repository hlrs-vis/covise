/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCGLOBALACTION_H
#define OSCGLOBALACTION_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscSetEnvironment.h"
#include "oscActionEntity.h"
#include "oscActionParameter.h"
#include "oscInfrastructure.h"
#include "oscTraffic.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscGlobalAction : public oscObjectBase
{
public:
oscGlobalAction()
{
        OSC_OBJECT_ADD_MEMBER(SetEnvironment, "oscSetEnvironment", 1);
        OSC_OBJECT_ADD_MEMBER(Entity, "oscActionEntity", 1);
        OSC_OBJECT_ADD_MEMBER(Parameter, "oscActionParameter", 1);
        OSC_OBJECT_ADD_MEMBER(Infrastructure, "oscInfrastructure", 1);
        OSC_OBJECT_ADD_MEMBER(Traffic, "oscTraffic", 1);
    };
        const char *getScope(){return "";};
    oscSetEnvironmentMember SetEnvironment;
    oscActionEntityMember Entity;
    oscActionParameterMember Parameter;
    oscInfrastructureMember Infrastructure;
    oscTrafficMember Traffic;

};

typedef oscObjectVariable<oscGlobalAction *> oscGlobalActionMember;
typedef oscObjectVariableArray<oscGlobalAction *> oscGlobalActionArrayMember;


}

#endif //OSCGLOBALACTION_H
