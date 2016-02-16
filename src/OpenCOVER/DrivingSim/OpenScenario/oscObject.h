/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_OBJECT_H
#define OSC_OBJECT_H

#include "oscExport.h"
#include "oscNameUserData.h"
#include "oscObjectVariable.h"

#include "oscCatalogReferenceTypeA.h"
#include "oscPosition.h"
#include "oscInitDynamics.h"
#include "oscInitState.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscObject: public oscNameUserData
{
public:
    oscObject()
    {
        OSC_OBJECT_ADD_MEMBER(catalogReference, "oscCatalogReferenceTypeA");
        OSC_OBJECT_ADD_MEMBER(initPosition, "oscPosition");
        OSC_OBJECT_ADD_MEMBER(initDynamics, "oscInitDynamics");
        OSC_OBJECT_ADD_MEMBER(initState, "oscInitState");
    };
    
    oscCatalogReferenceTypeAMember catalogReference;
    oscPositionMember initPosition;
    oscInitDynamicsMember initDynamics;
    oscInitStateMember initState;
};

typedef oscObjectVariable<oscObject *> oscObjectMember;

}

#endif //OSC_OBJECT_H
