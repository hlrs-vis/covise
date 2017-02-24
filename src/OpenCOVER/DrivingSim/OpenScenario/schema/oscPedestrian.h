/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCPEDESTRIAN_H
#define OSCPEDESTRIAN_H

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
class OPENSCENARIOEXPORT Enum_Pedestrian_categoryType : public oscEnumType
{
public:
static Enum_Pedestrian_categoryType *instance();
    private:
		Enum_Pedestrian_categoryType();
	    static Enum_Pedestrian_categoryType *inst; 
};
class OPENSCENARIOEXPORT oscPedestrian : public oscObjectBase
{
public:
oscPedestrian()
{
        OSC_ADD_MEMBER(model, 0);
        OSC_ADD_MEMBER(mass, 0);
        OSC_ADD_MEMBER(name, 0);
        OSC_ADD_MEMBER(category, 0);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(ParameterDeclaration, "oscParameterDeclaration", 0);
        OSC_OBJECT_ADD_MEMBER(BoundingBox, "oscBoundingBox", 0);
        OSC_OBJECT_ADD_MEMBER(Properties, "oscProperties", 0);
        category.enumType = Enum_Pedestrian_categoryType::instance();
    };
        const char *getScope(){return "";};
    oscString model;
    oscDouble mass;
    oscString name;
    oscEnum category;
    oscParameterDeclarationMember ParameterDeclaration;
    oscBoundingBoxMember BoundingBox;
    oscPropertiesMember Properties;

    enum Enum_Pedestrian_category
    {
pedestrian,
wheelchair,
animal,

    };

};

typedef oscObjectVariable<oscPedestrian *> oscPedestrianMember;
typedef oscObjectVariableArray<oscPedestrian *> oscPedestrianArrayMember;


}

#endif //OSCPEDESTRIAN_H
