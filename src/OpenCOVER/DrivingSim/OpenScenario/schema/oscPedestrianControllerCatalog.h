/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCPEDESTRIANCONTROLLERCATALOG_H
#define OSCPEDESTRIANCONTROLLERCATALOG_H

#include "oscExport.h"
#include "oscCatalog.h"
#include "oscObjectVariable.h"
#include "oscObjectVariableArray.h"

#include "oscVariables.h"
#include "schema/oscDirectory.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscPedestrianControllerCatalog : public oscCatalog
{
public:
oscPedestrianControllerCatalog()
{
}

};

typedef oscObjectVariable<oscPedestrianControllerCatalog *> oscPedestrianControllerCatalogMember;
typedef oscObjectVariableArray<oscPedestrianControllerCatalog *> oscPedestrianControllerCatalogArrayMember;


}

#endif //OSCPEDESTRIANCONTROLLERCATALOG_H
