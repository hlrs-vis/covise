/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCROUTECATALOG_H
#define OSCROUTECATALOG_H

#include "../oscExport.h"
#include "../oscCatalog.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscDirectory.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscRouteCatalog : public oscCatalog
{
public:
oscRouteCatalog()
{
}

};

typedef oscObjectVariable<oscRouteCatalog *> oscRouteCatalogMember;
typedef oscObjectVariableArray<oscRouteCatalog *> oscRouteCatalogArrayMember;


}

#endif //OSCROUTECATALOG_H
