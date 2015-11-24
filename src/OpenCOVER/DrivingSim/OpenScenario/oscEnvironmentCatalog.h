/* This EnvironmentCatalog is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_ENVIRONMENT_CATALOG_H
#define OSC_ENVIRONMENT_CATALOG_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>
#include <oscCatalog.h>

namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscEnvironmentCatalog: public oscObjectBase
{
public:
    oscEnvironmentCatalog()
    {
        OSC_OBJECT_ADD_MEMBER(environment,"oscCatalog");
    };
    oscCatalogMember environment;
};

typedef oscObjectVariable<oscEnvironmentCatalog *> oscEnvironmentCatalogMember;

}

#endif //OSC_ENVIRONMENT_CATALOG_H
