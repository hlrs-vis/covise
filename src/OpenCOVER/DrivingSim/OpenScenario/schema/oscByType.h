/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCBYTYPE_H
#define OSCBYTYPE_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT Enum_ObjectTypeType : public oscEnumType
{
public:
static Enum_ObjectTypeType *instance();
    private:
		Enum_ObjectTypeType();
	    static Enum_ObjectTypeType *inst; 
};
class OPENSCENARIOEXPORT oscByType : public oscObjectBase
{
public:
oscByType()
{
        OSC_ADD_MEMBER(type);
        type.enumType = Enum_ObjectTypeType::instance();
    };
    oscEnum type;

    enum Enum_ObjectType
    {
pedestrian,
vehicle,
miscellaneous,

    };

};

typedef oscObjectVariable<oscByType *> oscByTypeMember;
typedef oscObjectVariableArray<oscByType *> oscByTypeArrayMember;


}

#endif //OSCBYTYPE_H
