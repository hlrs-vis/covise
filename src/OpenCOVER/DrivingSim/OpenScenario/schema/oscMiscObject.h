/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCMISCOBJECT_H
#define OSCMISCOBJECT_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscParameterDeclaration.h"
#include "oscBoundingBox.h"
#include "oscProperties.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT Enum_MiscObject_categoryType : public oscEnumType
{
public:
static Enum_MiscObject_categoryType *instance();
    private:
		Enum_MiscObject_categoryType();
	    static Enum_MiscObject_categoryType *inst; 
};
class OPENSCENARIOEXPORT oscMiscObject : public oscObjectBase
{
public:
oscMiscObject()
{
        OSC_ADD_MEMBER(category, 0);
        OSC_ADD_MEMBER(mass, 0);
        OSC_ADD_MEMBER(name, 0);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(ParameterDeclaration, "oscParameterDeclaration", 0);
        OSC_OBJECT_ADD_MEMBER(BoundingBox, "oscBoundingBox", 0);
        OSC_OBJECT_ADD_MEMBER(Properties, "oscProperties", 0);
        category.enumType = Enum_MiscObject_categoryType::instance();
    };
        const char *getScope(){return "";};
    oscEnum category;
    oscDouble mass;
    oscString name;
    oscParameterDeclarationMember ParameterDeclaration;
    oscBoundingBoxMember BoundingBox;
    oscPropertiesMember Properties;

    enum Enum_MiscObject_category
    {
barrier,
guardRail,
other,

    };

};

typedef oscObjectVariable<oscMiscObject *> oscMiscObjectMember;
typedef oscObjectVariableArray<oscMiscObject *> oscMiscObjectArrayMember;


}

#endif //OSCMISCOBJECT_H
