/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCOBJECT_H
#define OSCOBJECT_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscCatalogReference.h"
#include "oscVehicle.h"
#include "oscPedestrian.h"
#include "oscMiscObject.h"
#include "oscObjectController.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscObject : public oscObjectBase
{
public:
oscObject()
{
        OSC_ADD_MEMBER(name, 0);
        OSC_OBJECT_ADD_MEMBER(CatalogReference, "oscCatalogReference", 1);
        OSC_OBJECT_ADD_MEMBER(Vehicle, "oscVehicle", 1);
        OSC_OBJECT_ADD_MEMBER(Pedestrian, "oscPedestrian", 1);
        OSC_OBJECT_ADD_MEMBER(MiscObject, "oscMiscObject", 1);
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Controller, "oscObjectController", 0);
    };
        const char *getScope(){return "/OpenSCENARIO/Entities";};
    oscString name;
    oscCatalogReferenceMember CatalogReference;
    oscVehicleMember Vehicle;
    oscPedestrianMember Pedestrian;
    oscMiscObjectMember MiscObject;
    oscObjectControllerMember Controller;

};

typedef oscObjectVariable<oscObject *> oscObjectMember;
typedef oscObjectVariableArray<oscObject *> oscObjectArrayMember;


}

#endif //OSCOBJECT_H
