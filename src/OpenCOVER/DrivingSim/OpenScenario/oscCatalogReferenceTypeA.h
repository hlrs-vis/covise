/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_CATALOG_REFERENCE_TYPE_A_H
#define OSC_CATALOG_REFERENCE_TYPE_A_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "oscParameterListTypeB.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscCatalogReferenceTypeA: public oscObjectBase
{
public:
    oscCatalogReferenceTypeA()
    {
        OSC_ADD_MEMBER(catalogId);
        OSC_OBJECT_ADD_MEMBER(parameterList, "oscParameterListTypeB");
    };

    oscString catalogId;
    oscParameterListTypeBArrayMember parameterList;
};

typedef oscObjectVariable<oscCatalogReferenceTypeA *> oscCatalogReferenceTypeAMember;

}

#endif //OSC_CATALOG_REFERENCE_TYPE_A_H
