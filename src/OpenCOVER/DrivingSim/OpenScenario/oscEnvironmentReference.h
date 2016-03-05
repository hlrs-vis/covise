/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_ENVIRONMENT_REFERENCE_H
#define OSC_ENVIRONMENT_REFERENCE_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscCatalogReferenceTypeA.h"
#include "oscUserDataList.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscEnvironmentReference: public oscObjectBase
{

public:
    oscEnvironmentReference()
    {
        OSC_OBJECT_ADD_MEMBER(catalogReference, "oscCatalogReferenceTypeA");
        OSC_OBJECT_ADD_MEMBER(userDataList, "oscUserDataList");
    };

    oscCatalogReferenceTypeAMember catalogReference;
    oscUserDataListMemberArray userDataList;
};

typedef oscObjectVariable<oscEnvironmentReference *> oscEnvironmentReferenceMember;

}

#endif //OSC_ENVIRONMENT_REFERENCE_H
