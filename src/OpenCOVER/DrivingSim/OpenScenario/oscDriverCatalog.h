/* This DriverCatalog is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_DRIVER_CATALOG_H
#define OSC_DRIVER_CATALOG_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscDirectory.h"
#include "oscUserDataList.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscDriverCatalog: public oscObjectBase
{
public:
    oscDriverCatalog()
    {
        OSC_OBJECT_ADD_MEMBER(directory, "oscDirectory");
        OSC_OBJECT_ADD_MEMBER(userDataList, "oscUserDataList");
    };

    oscDirectoryMember directory;
    oscUserDataListArrayMember userDataList;
};

typedef oscObjectVariable<oscDriverCatalog *> oscDriverCatalogMember;

}

#endif //OSC_DRIVER_CATALOG_H
