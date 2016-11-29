/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCOPENSCENARIO_VEHICLECATALOG_H
#define OSCOPENSCENARIO_VEHICLECATALOG_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscFileHeader.h"
#include "schema/oscVehicle.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscOpenSCENARIO_VehicleCatalog : public oscObjectBase
{
public:
    oscOpenSCENARIO_VehicleCatalog()
    {
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(FileHeader, "oscFileHeader");
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Vehicle, "oscVehicle");
    };
    oscFileHeaderMember FileHeader;
    oscVehicleMember Vehicle;

};

typedef oscObjectVariable<oscOpenSCENARIO_VehicleCatalog *> oscOpenSCENARIO_VehicleCatalogMember;


}

#endif //OSCOPENSCENARIO_VEHICLECATALOG_H
