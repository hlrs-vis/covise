/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_FILTERS_H
#define OSC_FILTERS_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariableArray.h"

#include "oscFilter.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscFilters: public oscObjectBase
{
public:
    oscFilters()
    {
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(filter, "oscFilter");
    };

    oscFilterMember filter;
};

typedef oscObjectVariableArray<oscFilters *> oscFiltersArrayMember;

}

#endif /* OSC_FILTERS_H */
