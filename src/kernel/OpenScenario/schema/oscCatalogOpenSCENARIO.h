/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCCATALOGOPENSCENARIO_H
#define OSCCATALOGOPENSCENARIO_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscFileHeader.h"
#include "oscCatalogObject.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscCatalogOpenSCENARIO : public oscObjectBase
{
public:
oscCatalogOpenSCENARIO()
{
        OSC_OBJECT_ADD_MEMBER(FileHeader, "oscFileHeader", 0);
        OSC_OBJECT_ADD_MEMBER(Catalog, "oscCatalogObject", 0);
    };
        const char *getScope(){return "/Catalog";};
    oscFileHeaderMember FileHeader;
    oscCatalogObjectMember Catalog;

};

typedef oscObjectVariable<oscCatalogOpenSCENARIO *> oscCatalogOpenSCENARIOMember;
typedef oscObjectVariableArray<oscCatalogOpenSCENARIO *> oscCatalogOpenSCENARIOArrayMember;


}

#endif //OSCCATALOGOPENSCENARIO_H
