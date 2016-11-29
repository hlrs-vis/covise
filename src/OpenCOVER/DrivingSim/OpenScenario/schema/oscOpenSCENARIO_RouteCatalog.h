/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCOPENSCENARIO_ROUTECATALOG_H
#define OSCOPENSCENARIO_ROUTECATALOG_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscFileHeader.h"
#include "schema/oscRoute.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscOpenSCENARIO_RouteCatalog : public oscObjectBase
{
public:
    oscOpenSCENARIO_RouteCatalog()
    {
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(FileHeader, "oscFileHeader");
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Route, "oscRoute");
    };
    oscFileHeaderMember FileHeader;
    oscRouteMember Route;

};

typedef oscObjectVariable<oscOpenSCENARIO_RouteCatalog *> oscOpenSCENARIO_RouteCatalogMember;


}

#endif //OSCOPENSCENARIO_ROUTECATALOG_H
