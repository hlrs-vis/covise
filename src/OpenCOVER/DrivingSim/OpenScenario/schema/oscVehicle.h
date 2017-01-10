/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCVEHICLE_H
#define OSCVEHICLE_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscBoundingBox.h"
#include "oscPerformance.h"
#include "oscAxles.h"
#include "oscParameterList.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT Enum_Vehicle_categoryType : public oscEnumType
{
public:
static Enum_Vehicle_categoryType *instance();
    private:
		Enum_Vehicle_categoryType();
	    static Enum_Vehicle_categoryType *inst; 
};
class OPENSCENARIOEXPORT oscVehicle : public oscObjectBase
{
public:
oscVehicle()
{
        OSC_ADD_MEMBER(name);
        OSC_ADD_MEMBER(category);
        OSC_OBJECT_ADD_MEMBER(BoundingBox, "oscBoundingBox");
        OSC_OBJECT_ADD_MEMBER(Performance, "oscPerformance");
        OSC_OBJECT_ADD_MEMBER(Axles, "oscAxles");
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(OSCParameterList, "oscParameterList");
        category.enumType = Enum_Vehicle_categoryType::instance();
    };
    oscString name;
    oscEnum category;
    oscBoundingBoxMember BoundingBox;
    oscPerformanceMember Performance;
    oscAxlesMember Axles;
    oscParameterListMember OSCParameterList;

    enum Enum_Vehicle_category
    {
car,
van,
truck,
trailer,
semitrailer,
bus,
motorbike,
bicycle,
train,
tram,

    };

};

typedef oscObjectVariable<oscVehicle *> oscVehicleMember;
typedef oscObjectVariableArray<oscVehicle *> oscVehicleArrayMember;


}

#endif //OSCVEHICLE_H
