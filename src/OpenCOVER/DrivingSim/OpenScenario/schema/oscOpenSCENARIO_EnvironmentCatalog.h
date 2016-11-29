/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCOPENSCENARIO_ENVIRONMENTCATALOG_H
#define OSCOPENSCENARIO_ENVIRONMENTCATALOG_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscFileHeader.h"
#include "schema/oscEnvironment.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscOpenSCENARIO_EnvironmentCatalog : public oscObjectBase
{
public:
    oscOpenSCENARIO_EnvironmentCatalog()
    {
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(FileHeader, "oscFileHeader");
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Environment, "oscEnvironment");
    };
    oscFileHeaderMember FileHeader;
    oscEnvironmentMember Environment;

};

typedef oscObjectVariable<oscOpenSCENARIO_EnvironmentCatalog *> oscOpenSCENARIO_EnvironmentCatalogMember;


}

#endif //OSCOPENSCENARIO_ENVIRONMENTCATALOG_H
