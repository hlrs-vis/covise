/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCLATERALPARAMS_H
#define OSCLATERALPARAMS_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT Enum_Lateral_purposeType : public oscEnumType
{
public:
static Enum_Lateral_purposeType *instance();
    private:
		Enum_Lateral_purposeType();
	    static Enum_Lateral_purposeType *inst; 
};
class OPENSCENARIOEXPORT oscLateralParams : public oscObjectBase
{
public:
oscLateralParams()
{
        OSC_ADD_MEMBER(purpose, 0);
        purpose.enumType = Enum_Lateral_purposeType::instance();
    };
        const char *getScope(){return "/OSCPrivateAction/Routing/FollowTrajectory";};
    oscEnum purpose;

    enum Enum_Lateral_purpose
    {
position,
steering,

    };

};

typedef oscObjectVariable<oscLateralParams *> oscLateralParamsMember;
typedef oscObjectVariableArray<oscLateralParams *> oscLateralParamsArrayMember;


}

#endif //OSCLATERALPARAMS_H
