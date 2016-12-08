/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCPEDESTRIANCATALOG_H
#define OSCPEDESTRIANCATALOG_H

#include "oscExport.h"
#include "oscCatalog.h"
#include "oscObjectVariable.h"
#include "oscObjectVariableArray.h"

#include "oscVariables.h"
#include "schema/oscDirectory.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscPedestrianCatalog : public oscCatalog
{
public:
oscPedestrianCatalog()
{
}

};

typedef oscObjectVariable<oscPedestrianCatalog *> oscPedestrianCatalogMember;
typedef oscObjectVariableArray<oscPedestrianCatalog *> oscPedestrianCatalogArrayMember;


}

#endif //OSCPEDESTRIANCATALOG_H
