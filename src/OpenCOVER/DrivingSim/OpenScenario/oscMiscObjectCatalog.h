/* This MiscObjectCatalog is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_MISC_OBJECT_CATALOG_H
#define OSC_MISC_OBJECT_CATALOG_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>
#include <oscCatalog.h>

namespace OpenScenario {

class OpenScenarioBase;

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscMiscObjectCatalog: public oscObjectBase
{
public:
    oscMiscObjectCatalog()
    {
        OSC_OBJECT_ADD_MEMBER(miscObject,"oscCatalog");
    };
    oscCatalogMember miscObject;
};

typedef oscObjectVariable<oscMiscObjectCatalog *> oscMiscObjectCatalogMember;

}

#endif //OSC_MISC_OBJECT_CATALOG_H
