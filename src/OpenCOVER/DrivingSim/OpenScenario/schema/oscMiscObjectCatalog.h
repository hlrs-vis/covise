/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCMISCOBJECTCATALOG_H
#define OSCMISCOBJECTCATALOG_H

#include "oscExport.h"
#include "oscCatalog.h"
#include "oscObjectVariable.h"
#include "oscObjectVariableArray.h"

#include "oscVariables.h"
#include "schema/oscDirectory.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscMiscObjectCatalog : public oscCatalog
{
public:
oscMiscObjectCatalog()
{
}

};

typedef oscObjectVariable<oscMiscObjectCatalog *> oscMiscObjectCatalogMember;
typedef oscObjectVariableArray<oscMiscObjectCatalog *> oscMiscObjectCatalogArrayMember;


}

#endif //OSCMISCOBJECTCATALOG_H
