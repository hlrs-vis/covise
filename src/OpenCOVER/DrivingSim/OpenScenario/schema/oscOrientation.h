/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCORIENTATION_H
#define OSCORIENTATION_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT Enum_Orientation_typeType : public oscEnumType
{
public:
static Enum_Orientation_typeType *instance();
    private:
		Enum_Orientation_typeType();
	    static Enum_Orientation_typeType *inst; 
};
class OPENSCENARIOEXPORT oscOrientation : public oscObjectBase
{
public:
oscOrientation()
{
        OSC_ADD_MEMBER_OPTIONAL(type, 0);
        OSC_ADD_MEMBER_OPTIONAL(h, 0);
        OSC_ADD_MEMBER_OPTIONAL(p, 0);
        OSC_ADD_MEMBER_OPTIONAL(r, 0);
        type.enumType = Enum_Orientation_typeType::instance();
    };
        const char *getScope(){return "";};
    oscEnum type;
    oscDouble h;
    oscDouble p;
    oscDouble r;

    enum Enum_Orientation_type
    {
relative,
absolute,

    };

};

typedef oscObjectVariable<oscOrientation *> oscOrientationMember;
typedef oscObjectVariableArray<oscOrientation *> oscOrientationArrayMember;


}

#endif //OSCORIENTATION_H
