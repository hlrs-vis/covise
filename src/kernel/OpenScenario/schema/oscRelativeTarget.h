/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCRELATIVETARGET_H
#define OSCRELATIVETARGET_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT Enum_Speed_Target_valueTypeType : public oscEnumType
{
public:
static Enum_Speed_Target_valueTypeType *instance();
    private:
		Enum_Speed_Target_valueTypeType();
	    static Enum_Speed_Target_valueTypeType *inst; 
};
class OPENSCENARIOEXPORT oscRelativeTarget : public oscObjectBase
{
public:
oscRelativeTarget()
{
        OSC_ADD_MEMBER(object, 0);
        OSC_ADD_MEMBER(value, 0);
        OSC_ADD_MEMBER(valueType, 0);
        OSC_ADD_MEMBER(continuous, 0);
        valueType.enumType = Enum_Speed_Target_valueTypeType::instance();
    };
        const char *getScope(){return "/OSCPrivateAction/Longitudinal/Speed/Target";};
    oscString object;
    oscDouble value;
    oscEnum valueType;
    oscBool continuous;

    enum Enum_Speed_Target_valueType
    {
delta,
factor,

    };

};

typedef oscObjectVariable<oscRelativeTarget *> oscRelativeTargetMember;
typedef oscObjectVariableArray<oscRelativeTarget *> oscRelativeTargetArrayMember;


}

#endif //OSCRELATIVETARGET_H
