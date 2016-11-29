/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCVEHICLECATALOG_H
#define OSCVEHICLECATALOG_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscDirectory.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscVehicleCatalog : public oscObjectBase
{
public:
    oscVehicleCatalog()
    {
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Directory, "oscDirectory");
    };
    oscDirectoryMember Directory;

};

typedef oscObjectVariable<oscVehicleCatalog *> oscVehicleCatalogMember;


}

#endif //OSCVEHICLECATALOG_H
