/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCGLOBALACTION_H
#define OSCGLOBALACTION_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscEnvironment.h"
#include "schema/oscEntity.h"
#include "schema/oscActionParameter.h"
#include "schema/oscInfrastructure.h"
#include "schema/oscTraffic.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscGlobalAction : public oscObjectBase
{
public:
    oscGlobalAction()
    {
        OSC_OBJECT_ADD_MEMBER(Environment, "oscEnvironment");
        OSC_OBJECT_ADD_MEMBER(Entity, "oscEntity");
        OSC_OBJECT_ADD_MEMBER(ActionParameter, "oscActionParameter");
        OSC_OBJECT_ADD_MEMBER(Infrastructure, "oscInfrastructure");
        OSC_OBJECT_ADD_MEMBER(Traffic, "oscTraffic");
    };
    oscEnvironmentMember Environment;
    oscEntityMember Entity;
    oscActionParameterMember ActionParameter;
    oscInfrastructureMember Infrastructure;
    oscTrafficMember Traffic;

};

typedef oscObjectVariable<oscGlobalAction *> oscGlobalActionMember;


}

#endif //OSCGLOBALACTION_H
