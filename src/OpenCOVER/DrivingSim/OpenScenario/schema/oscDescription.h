/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCDESCRIPTION_H
#define OSCDESCRIPTION_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscParameter.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT Enum_sexType : public oscEnumType
{
public:
static Enum_sexType *instance();
    private:
		Enum_sexType();
	    static Enum_sexType *inst; 
};
class OPENSCENARIOEXPORT oscDescription : public oscObjectBase
{
public:
oscDescription()
{
        OSC_ADD_MEMBER(weight, 0);
        OSC_ADD_MEMBER(height, 0);
        OSC_ADD_MEMBER(eyeDistance, 0);
        OSC_ADD_MEMBER(age, 0);
        OSC_ADD_MEMBER(sex, 0);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Parameter, "oscParameter", 0);
        sex.enumType = Enum_sexType::instance();
    };
    oscDouble weight;
    oscDouble height;
    oscDouble eyeDistance;
    oscDouble age;
    oscEnum sex;
    oscParameterArrayMember Parameter;

    enum Enum_sex
    {
male,
female,

    };

};

typedef oscObjectVariable<oscDescription *> oscDescriptionMember;
typedef oscObjectVariableArray<oscDescription *> oscDescriptionArrayMember;


}

#endif //OSCDESCRIPTION_H
