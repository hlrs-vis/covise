/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCMANEUVERCATALOG_H
#define OSCMANEUVERCATALOG_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscDirectory.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscManeuverCatalog : public oscObjectBase
{
public:
    oscManeuverCatalog()
    {
        OSC_OBJECT_ADD_MEMBER(Directory, "oscDirectory");
    };
    oscDirectoryMember Directory;

};

typedef oscObjectVariable<oscManeuverCatalog *> oscManeuverCatalogMember;


}

#endif //OSCMANEUVERCATALOG_H
