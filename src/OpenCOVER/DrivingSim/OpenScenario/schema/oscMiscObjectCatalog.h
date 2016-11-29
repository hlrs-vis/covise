/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCMISCOBJECTCATALOG_H
#define OSCMISCOBJECTCATALOG_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscDirectory.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscMiscObjectCatalog : public oscObjectBase
{
public:
    oscMiscObjectCatalog()
    {
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Directory, "oscDirectory");
    };
    oscDirectoryMember Directory;

};

typedef oscObjectVariable<oscMiscObjectCatalog *> oscMiscObjectCatalogMember;


}

#endif //OSCMISCOBJECTCATALOG_H
