/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_FILTER_H
#define OSC_FILTER_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>

namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscFilter: public oscObjectBase
{
public:
    oscFilter()
    {
        OSC_ADD_MEMBER(ObjectType);
		OSC_ADD_MEMBER(maxNum);
		OSC_ADD_MEMBER(filterParam);
    };
    oscString ObjectType;
	oscInt maxNum;
	oscDouble filterParam;
};

typedef oscObjectVariable<oscFilter *> oscFilterMember;

}

#endif //OSC_FILTER_H
