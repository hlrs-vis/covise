/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCENVIRONMENTCATALOG_H
#define OSCENVIRONMENTCATALOG_H

#include "../oscExport.h"
#include "../oscCatalog.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscDirectory.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscEnvironmentCatalog : public oscCatalog
{
public:
oscEnvironmentCatalog()
{
}

};

typedef oscObjectVariable<oscEnvironmentCatalog *> oscEnvironmentCatalogMember;
typedef oscObjectVariableArray<oscEnvironmentCatalog *> oscEnvironmentCatalogArrayMember;


}

#endif //OSCENVIRONMENTCATALOG_H
