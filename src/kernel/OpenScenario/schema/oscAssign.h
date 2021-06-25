/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCASSIGN_H
#define OSCASSIGN_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscDriver.h"
#include "oscPedestrianController.h"
#include "oscCatalogReference.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscAssign : public oscObjectBase
{
public:
oscAssign()
{
        OSC_OBJECT_ADD_MEMBER(Driver, "oscDriver", 1);
        OSC_OBJECT_ADD_MEMBER(PedestrianController, "oscPedestrianController", 1);
        OSC_OBJECT_ADD_MEMBER(CatalogReference, "oscCatalogReference", 1);
    };
        const char *getScope(){return "/OSCPrivateAction/ActionController";};
    oscDriverMember Driver;
    oscPedestrianControllerMember PedestrianController;
    oscCatalogReferenceMember CatalogReference;

};

typedef oscObjectVariable<oscAssign *> oscAssignMember;
typedef oscObjectVariableArray<oscAssign *> oscAssignArrayMember;


}

#endif //OSCASSIGN_H
