/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCOPENSCENARIO_MANEUVERCATALOG_H
#define OSCOPENSCENARIO_MANEUVERCATALOG_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"
#include "oscObjectVariableArray.h"

#include "oscVariables.h"
#include "schema/oscFileHeader.h"
#include "schema/oscManeuver.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscOpenSCENARIO_ManeuverCatalog : public oscObjectBase
{
public:
oscOpenSCENARIO_ManeuverCatalog()
{
        OSC_OBJECT_ADD_MEMBER(FileHeader, "oscFileHeader");
        OSC_OBJECT_ADD_MEMBER(Maneuver, "oscManeuver");
    };
    oscFileHeaderMember FileHeader;
    oscManeuverArrayMember Maneuver;

};

typedef oscObjectVariable<oscOpenSCENARIO_ManeuverCatalog *> oscOpenSCENARIO_ManeuverCatalogMember;
typedef oscObjectVariableArray<oscOpenSCENARIO_ManeuverCatalog *> oscOpenSCENARIO_ManeuverCatalogArrayMember;


}

#endif //OSCOPENSCENARIO_MANEUVERCATALOG_H
