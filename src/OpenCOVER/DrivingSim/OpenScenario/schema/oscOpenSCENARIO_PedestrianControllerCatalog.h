/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCOPENSCENARIO_PEDESTRIANCONTROLLERCATALOG_H
#define OSCOPENSCENARIO_PEDESTRIANCONTROLLERCATALOG_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscFileHeader.h"
#include "schema/oscPedestrianController.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscOpenSCENARIO_PedestrianControllerCatalog : public oscObjectBase
{
public:
    oscOpenSCENARIO_PedestrianControllerCatalog()
    {
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(FileHeader, "oscFileHeader");
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(PedestrianController, "oscPedestrianController");
    };
    oscFileHeaderMember FileHeader;
    oscPedestrianControllerMember PedestrianController;

};

typedef oscObjectVariable<oscOpenSCENARIO_PedestrianControllerCatalog *> oscOpenSCENARIO_PedestrianControllerCatalogMember;


}

#endif //OSCOPENSCENARIO_PEDESTRIANCONTROLLERCATALOG_H
