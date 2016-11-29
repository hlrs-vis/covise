/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCVEHICLE_H
#define OSCVEHICLE_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscBoundingBox.h"
#include "schema/oscPerformance.h"
#include "schema/oscAxles.h"
#include "schema/oscParameterList.h"

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
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(BoundingBox, "oscBoundingBox");
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Performance, "oscPerformance");
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Axles, "oscAxles");
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(OSCParameterList, "oscParameterList");
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


}

#endif //OSCVEHICLE_H
