/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCATSTART_H
#define OSCATSTART_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"
#include "oscObjectVariableArray.h"

#include "oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT Enum_Story_Element_typeType : public oscEnumType
{
public:
static Enum_Story_Element_typeType *instance();
    private:
		Enum_Story_Element_typeType();
	    static Enum_Story_Element_typeType *inst; 
};
class OPENSCENARIOEXPORT oscAtStart : public oscObjectBase
{
public:
oscAtStart()
{
        OSC_ADD_MEMBER(type);
        OSC_ADD_MEMBER(name);
        type.enumType = Enum_Story_Element_typeType::instance();
    };
    oscEnum type;
    oscString name;

    enum Enum_Story_Element_type
    {
act,
scene,
maneuver,
event,
action,

    };

};

typedef oscObjectVariable<oscAtStart *> oscAtStartMember;
typedef oscObjectVariableArray<oscAtStart *> oscAtStartArrayMember;


}

#endif //OSCATSTART_H
