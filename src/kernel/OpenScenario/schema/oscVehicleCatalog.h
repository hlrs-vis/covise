/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCVEHICLECATALOG_H
#define OSCVEHICLECATALOG_H

#include "../oscExport.h"
#include "../oscCatalog.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscDirectory.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscVehicleCatalog : public oscCatalog
{
public:
oscVehicleCatalog()
{
}

};

typedef oscObjectVariable<oscVehicleCatalog *> oscVehicleCatalogMember;
typedef oscObjectVariableArray<oscVehicleCatalog *> oscVehicleCatalogArrayMember;


}

#endif //OSCVEHICLECATALOG_H
