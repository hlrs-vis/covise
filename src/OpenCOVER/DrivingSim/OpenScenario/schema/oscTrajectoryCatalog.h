/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCTRAJECTORYCATALOG_H
#define OSCTRAJECTORYCATALOG_H

#include "../oscExport.h"
#include "../oscCatalog.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscDirectory.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscTrajectoryCatalog : public oscCatalog
{
public:
oscTrajectoryCatalog()
{
}

};

typedef oscObjectVariable<oscTrajectoryCatalog *> oscTrajectoryCatalogMember;
typedef oscObjectVariableArray<oscTrajectoryCatalog *> oscTrajectoryCatalogArrayMember;


}

#endif //OSCTRAJECTORYCATALOG_H
