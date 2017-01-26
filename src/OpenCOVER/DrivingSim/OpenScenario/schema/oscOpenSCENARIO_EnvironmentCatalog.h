/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCOPENSCENARIO_ENVIRONMENTCATALOG_H
#define OSCOPENSCENARIO_ENVIRONMENTCATALOG_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscFileHeader.h"
#include "oscEnvironment.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscOpenSCENARIO_EnvironmentCatalog : public oscObjectBase
{
public:
oscOpenSCENARIO_EnvironmentCatalog()
{
        OSC_OBJECT_ADD_MEMBER(FileHeader, "oscFileHeader", 0);
        OSC_OBJECT_ADD_MEMBER(Environment, "oscEnvironment", 0);
    };
    oscFileHeaderMember FileHeader;
    oscEnvironmentArrayMember Environment;

};

typedef oscObjectVariable<oscOpenSCENARIO_EnvironmentCatalog *> oscOpenSCENARIO_EnvironmentCatalogMember;
typedef oscObjectVariableArray<oscOpenSCENARIO_EnvironmentCatalog *> oscOpenSCENARIO_EnvironmentCatalogArrayMember;


}

#endif //OSCOPENSCENARIO_ENVIRONMENTCATALOG_H
