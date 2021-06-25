/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCPERSONDESCRIPTION_H
#define OSCPERSONDESCRIPTION_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscProperties.h"

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
class OPENSCENARIOEXPORT oscPersonDescription : public oscObjectBase
{
public:
oscPersonDescription()
{
        OSC_ADD_MEMBER(weight, 0);
        OSC_ADD_MEMBER(height, 0);
        OSC_ADD_MEMBER(eyeDistance, 0);
        OSC_ADD_MEMBER(age, 0);
        OSC_ADD_MEMBER(sex, 0);
        OSC_OBJECT_ADD_MEMBER(Properties, "oscProperties", 0);
        sex.enumType = Enum_sexType::instance();
    };
        const char *getScope(){return "";};
    oscDouble weight;
    oscDouble height;
    oscDouble eyeDistance;
    oscDouble age;
    oscEnum sex;
    oscPropertiesMember Properties;

    enum Enum_sex
    {
male,
female,

    };

};

typedef oscObjectVariable<oscPersonDescription *> oscPersonDescriptionMember;
typedef oscObjectVariableArray<oscPersonDescription *> oscPersonDescriptionArrayMember;


}

#endif //OSCPERSONDESCRIPTION_H
