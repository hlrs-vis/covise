/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCOPENSCENARIO_MISCOBJECTCATALOG_H
#define OSCOPENSCENARIO_MISCOBJECTCATALOG_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscFileHeader.h"
#include "schema/oscMiscObject.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscOpenSCENARIO_MiscObjectCatalog : public oscObjectBase
{
public:
    oscOpenSCENARIO_MiscObjectCatalog()
    {
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(FileHeader, "oscFileHeader");
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(MiscObject, "oscMiscObject");
    };
    oscFileHeaderMember FileHeader;
    oscMiscObjectMember MiscObject;

};

typedef oscObjectVariable<oscOpenSCENARIO_MiscObjectCatalog *> oscOpenSCENARIO_MiscObjectCatalogMember;


}

#endif //OSCOPENSCENARIO_MISCOBJECTCATALOG_H
