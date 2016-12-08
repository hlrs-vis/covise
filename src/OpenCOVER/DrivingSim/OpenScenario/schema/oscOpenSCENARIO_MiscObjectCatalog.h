/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCOPENSCENARIO_MISCOBJECTCATALOG_H
#define OSCOPENSCENARIO_MISCOBJECTCATALOG_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"
#include "oscObjectVariableArray.h"

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
        OSC_OBJECT_ADD_MEMBER(FileHeader, "oscFileHeader");
        OSC_OBJECT_ADD_MEMBER(MiscObject, "oscMiscObject");
    };
    oscFileHeaderMember FileHeader;
    oscMiscObjectArrayMember MiscObject;

};

typedef oscObjectVariable<oscOpenSCENARIO_MiscObjectCatalog *> oscOpenSCENARIO_MiscObjectCatalogMember;
typedef oscObjectVariableArray<oscOpenSCENARIO_MiscObjectCatalog *> oscOpenSCENARIO_MiscObjectCatalogArrayMember;


}

#endif //OSCOPENSCENARIO_MISCOBJECTCATALOG_H
