/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_OBSERVER_TYPE_A_H
#define OSC_OBSERVER_TYPE_A_H

#include "oscExport.h"
#include "oscNameRefIdUserData.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "oscFileHeader.h"
#include "oscFrustum.h"
#include "oscFilters.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscObserverTypeA: public oscNameRefIdUserData
{
public:
    oscObserverTypeA()
    {
        OSC_OBJECT_ADD_MEMBER(fileHeader, "oscFileHeader");
        OSC_ADD_MEMBER(type);
        OSC_OBJECT_ADD_MEMBER(frustum, "oscFrustum");
        OSC_OBJECT_ADD_MEMBER(filters, "oscFilters");
    };

    oscFileHeaderMember fileHeader;
    oscString type;
    oscFrustumMember frustum;
    oscFiltersArrayMember filters;
};

typedef oscObjectVariable<oscObserverTypeA *> oscObserverTypeAMember;

}

#endif //OSC_OBSERVER_TYPE_A_H
