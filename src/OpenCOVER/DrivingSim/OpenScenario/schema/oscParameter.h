/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCPARAMETER_H
#define OSCPARAMETER_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT Enum_OSC_Parameter_typeType : public oscEnumType
{
public:
static Enum_OSC_Parameter_typeType *instance();
    private:
		Enum_OSC_Parameter_typeType();
	    static Enum_OSC_Parameter_typeType *inst; 
};
class OPENSCENARIOEXPORT oscParameter : public oscObjectBase
{
public:
oscParameter()
{
        OSC_ADD_MEMBER(name, 0);
        OSC_ADD_MEMBER(type, 0);
        OSC_ADD_MEMBER(value, 0);
        type.enumType = Enum_OSC_Parameter_typeType::instance();
    };
        const char *getScope(){return "/OSCParameterDeclaration";};
    oscString name;
    oscEnum type;
    oscString value;

    enum Enum_OSC_Parameter_type
    {
int_t,
double_t,
string,

    };

};

typedef oscObjectVariable<oscParameter *> oscParameterMember;
typedef oscObjectVariableArray<oscParameter *> oscParameterArrayMember;


}

#endif //OSCPARAMETER_H
