/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCOBJECTCONTROLLER_H
#define OSCOBJECTCONTROLLER_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscCatalogReference.h"
#include "oscDriver.h"
#include "oscPedestrianController.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscObjectController : public oscObjectBase
{
public:
oscObjectController()
{
        OSC_OBJECT_ADD_MEMBER(CatalogReference, "oscCatalogReference", 1);
        OSC_OBJECT_ADD_MEMBER(Driver, "oscDriver", 1);
        OSC_OBJECT_ADD_MEMBER(PedestrianController, "oscPedestrianController", 1);
    };
        const char *getScope(){return "/OpenSCENARIO/Entities/Object";};
    oscCatalogReferenceMember CatalogReference;
    oscDriverMember Driver;
    oscPedestrianControllerMember PedestrianController;

};

typedef oscObjectVariable<oscObjectController *> oscObjectControllerMember;
typedef oscObjectVariableArray<oscObjectController *> oscObjectControllerArrayMember;


}

#endif //OSCOBJECTCONTROLLER_H
