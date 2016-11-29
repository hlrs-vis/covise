/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCPEDESTRIAN_H
#define OSCPEDESTRIAN_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscBoundingBox.h"

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
        OSC_ADD_MEMBER(model);
        OSC_ADD_MEMBER(mass);
        OSC_ADD_MEMBER(name);
        OSC_ADD_MEMBER(category);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(BoundingBox, "oscBoundingBox");
    };
    oscString model;
    oscDouble mass;
    oscString name;
    oscEnum category;
    oscBoundingBoxMember BoundingBox;

    enum Enum_Pedestrian_category
    {
pedestrian,
wheelchair,
animal,

    };

};

typedef oscObjectVariable<oscPedestrian *> oscPedestrianMember;


}

#endif //OSCPEDESTRIAN_H
