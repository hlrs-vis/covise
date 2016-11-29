/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCMISCOBJECT_H
#define OSCMISCOBJECT_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscBoundingBox.h"

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
        OSC_ADD_MEMBER(category);
        OSC_ADD_MEMBER(mass);
        OSC_ADD_MEMBER(name);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(BoundingBox, "oscBoundingBox");
    };
    oscEnum category;
    oscDouble mass;
    oscString name;
    oscBoundingBoxMember BoundingBox;

    enum Enum_MiscObject_category
    {
barrier,
guardRail,
other,

    };

};

typedef oscObjectVariable<oscMiscObject *> oscMiscObjectMember;


}

#endif //OSCMISCOBJECT_H
