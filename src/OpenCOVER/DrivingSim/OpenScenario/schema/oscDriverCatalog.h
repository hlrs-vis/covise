/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCDRIVERCATALOG_H
#define OSCDRIVERCATALOG_H

#include "../oscExport.h"
#include "../oscCatalog.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscDirectory.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscDriverCatalog : public oscCatalog
{
public:
oscDriverCatalog()
{
}

};

typedef oscObjectVariable<oscDriverCatalog *> oscDriverCatalogMember;
typedef oscObjectVariableArray<oscDriverCatalog *> oscDriverCatalogArrayMember;


}

#endif //OSCDRIVERCATALOG_H
