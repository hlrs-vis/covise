/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCMANEUVERCATALOG_H
#define OSCMANEUVERCATALOG_H

#include "../oscExport.h"
#include "../oscCatalog.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscDirectory.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscManeuverCatalog : public oscCatalog
{
public:
oscManeuverCatalog()
{
}

};

typedef oscObjectVariable<oscManeuverCatalog *> oscManeuverCatalogMember;
typedef oscObjectVariableArray<oscManeuverCatalog *> oscManeuverCatalogArrayMember;


}

#endif //OSCMANEUVERCATALOG_H
