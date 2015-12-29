/* This OBSERVERCatalog is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_OBSERVER_CATALOG_H
#define OSC_OBSERVER_CATALOG_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscDirectory.h>
#include <oscUserData.h>

namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscObserverCatalog: public oscObjectBase
{
public:
    oscObserverCatalog()
    {
        OSC_OBJECT_ADD_MEMBER(directory,"oscDirectory");
		OSC_OBJECT_ADD_MEMBER(userData,"oscUserData");
    };
	oscDirectoryMember directory;
    oscUserDataMember userData;
};

typedef oscObjectVariable<oscObserverCatalog *> oscObserverCatalogMember;

}

#endif //OSC_OBSERVER_CATALOG_H
