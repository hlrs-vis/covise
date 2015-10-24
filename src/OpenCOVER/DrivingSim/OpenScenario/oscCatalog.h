/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_CATALOG_H
#define OSC_CATALOG_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>
#include <oscDirectory.h>
#include <oscUserData.h>

namespace OpenScenario {

class OpenScenarioBase;

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscCatalog: public oscObjectBase
{
public:
    oscCatalog()
    {
        OSC_ADD_MEMBER(directory);
		OSC_ADD_MEMBER(userData);
    };
	oscDirectoryMember directory;
    oscUserDataMember userData;
};

typedef oscObjectVariable<oscCatalog *> oscCatalogMember;

}

#endif //OSC_CATALOG_H