/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCDRIVERCATALOG_H
#define OSCDRIVERCATALOG_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscDirectory.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscDriverCatalog : public oscObjectBase
{
public:
    oscDriverCatalog()
    {
        OSC_OBJECT_ADD_MEMBER(Directory, "oscDirectory");
    };
    oscDirectoryMember Directory;

};

typedef oscObjectVariable<oscDriverCatalog *> oscDriverCatalogMember;


}

#endif //OSCDRIVERCATALOG_H
