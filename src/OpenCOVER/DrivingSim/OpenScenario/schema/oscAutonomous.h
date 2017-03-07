/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCAUTONOMOUS_H
#define OSCAUTONOMOUS_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT Enum_Controller_domainType : public oscEnumType
{
public:
static Enum_Controller_domainType *instance();
    private:
		Enum_Controller_domainType();
	    static Enum_Controller_domainType *inst; 
};
class OPENSCENARIOEXPORT oscAutonomous : public oscObjectBase
{
public:
oscAutonomous()
{
        OSC_ADD_MEMBER(activate, 0);
        OSC_ADD_MEMBER(domain, 0);
        domain.enumType = Enum_Controller_domainType::instance();
    };
        const char *getScope(){return "/OSCPrivateAction";};
    oscBool activate;
    oscEnum domain;

    enum Enum_Controller_domain
    {
longitudinal,
lateral,
both,

    };

};

typedef oscObjectVariable<oscAutonomous *> oscAutonomousMember;
typedef oscObjectVariableArray<oscAutonomous *> oscAutonomousArrayMember;


}

#endif //OSCAUTONOMOUS_H
