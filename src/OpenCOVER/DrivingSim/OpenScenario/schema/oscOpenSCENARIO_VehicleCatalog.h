/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCOPENSCENARIO_VEHICLECATALOG_H
#define OSCOPENSCENARIO_VEHICLECATALOG_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscFileHeader.h"
#include "oscVehicle.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscOpenSCENARIO_VehicleCatalog : public oscObjectBase
{
public:
oscOpenSCENARIO_VehicleCatalog()
{
        OSC_OBJECT_ADD_MEMBER(FileHeader, "oscFileHeader", 0);
        OSC_OBJECT_ADD_MEMBER(Vehicle, "oscVehicle", 0);
    };
    oscFileHeaderMember FileHeader;
    oscVehicleArrayMember Vehicle;

};

typedef oscObjectVariable<oscOpenSCENARIO_VehicleCatalog *> oscOpenSCENARIO_VehicleCatalogMember;
typedef oscObjectVariableArray<oscOpenSCENARIO_VehicleCatalog *> oscOpenSCENARIO_VehicleCatalogArrayMember;


}

#endif //OSCOPENSCENARIO_VEHICLECATALOG_H
