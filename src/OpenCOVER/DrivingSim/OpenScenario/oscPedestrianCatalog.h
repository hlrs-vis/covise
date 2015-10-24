/* This PEDESTRIANCatalog is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_PEDESTRIAN_CATALOG_H
#define OSC_PEDESTRIAN_CATALOG_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>
#include <oscCatalog.h>

namespace OpenScenario {

class OpenScenarioBase;

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscPedestrianCatalog: public oscObjectBase
{
public:
    oscPedestrianCatalog()
    {
        OSC_ADD_MEMBER(pedestrian);
    };
    oscCatalogMember pedestrian;
};

typedef oscObjectVariable<oscPedestrianCatalog *> oscPedestrianCatalogMember;

}

#endif //OSC_PEDESTRIAN_CATALOG_H