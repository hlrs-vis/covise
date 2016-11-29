/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCPEDESTRIANCONTROLLERCATALOG_H
#define OSCPEDESTRIANCONTROLLERCATALOG_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscDirectory.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscPedestrianControllerCatalog : public oscObjectBase
{
public:
    oscPedestrianControllerCatalog()
    {
        OSC_OBJECT_ADD_MEMBER(Directory, "oscDirectory");
    };
    oscDirectoryMember Directory;

};

typedef oscObjectVariable<oscPedestrianControllerCatalog *> oscPedestrianControllerCatalogMember;


}

#endif //OSCPEDESTRIANCONTROLLERCATALOG_H
