/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_CATALOG_BASE_H
#define OSC_CATALOG_BASE_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariableCatalog.h"

#include "oscDirectory.h"
#include "oscUserDataList.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscCatalogBase: public oscObjectBase
{
public:
    oscCatalogBase()
    {
        OSC_OBJECT_ADD_MEMBER(directory, "oscDirectory");
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(userDataList, "oscUserDataList");
    };

    oscDirectoryMember directory;
    oscUserDataListMemberArray userDataList;
};

typedef oscObjectVariableCatalog<oscCatalogBase *> oscCatalogBaseMemberCatalog;

}

#endif /* OSC_CATALOG_BASE_H */
