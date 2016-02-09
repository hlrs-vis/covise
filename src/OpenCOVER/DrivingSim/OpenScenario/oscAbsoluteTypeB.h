/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_ABSOLUTE_TYPE_B_H_
#define OSC_ABSOLUTE_TYPE_B_H_

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscAbsoluteTypeB: public oscObjectBase
{
public:
    oscAbsoluteTypeB()
    {
        OSC_ADD_MEMBER(target);
    };

    oscDouble target;
};

typedef oscObjectVariable<oscAbsoluteTypeB *> oscAbsoluteTypeBMember;

}

#endif //OSC_ABSOLUTE_TYPE_B_H_
