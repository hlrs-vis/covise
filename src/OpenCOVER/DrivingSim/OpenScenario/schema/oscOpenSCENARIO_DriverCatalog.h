/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCOPENSCENARIO_DRIVERCATALOG_H
#define OSCOPENSCENARIO_DRIVERCATALOG_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscFileHeader.h"
#include "oscDriver.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscOpenSCENARIO_DriverCatalog : public oscObjectBase
{
public:
oscOpenSCENARIO_DriverCatalog()
{
        OSC_OBJECT_ADD_MEMBER(FileHeader, "oscFileHeader");
        OSC_OBJECT_ADD_MEMBER(Driver, "oscDriver");
    };
    oscFileHeaderMember FileHeader;
    oscDriverArrayMember Driver;

};

typedef oscObjectVariable<oscOpenSCENARIO_DriverCatalog *> oscOpenSCENARIO_DriverCatalogMember;
typedef oscObjectVariableArray<oscOpenSCENARIO_DriverCatalog *> oscOpenSCENARIO_DriverCatalogArrayMember;


}

#endif //OSCOPENSCENARIO_DRIVERCATALOG_H
