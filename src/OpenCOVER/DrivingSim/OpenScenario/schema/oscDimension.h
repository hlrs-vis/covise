/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCDIMENSION_H
#define OSCDIMENSION_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscDimension : public oscObjectBase
{
public:
oscDimension()
{
        OSC_ADD_MEMBER(width);
        OSC_ADD_MEMBER(length);
        OSC_ADD_MEMBER(height);
    };
    oscDouble width;
    oscDouble length;
    oscDouble height;

};

typedef oscObjectVariable<oscDimension *> oscDimensionMember;
typedef oscObjectVariableArray<oscDimension *> oscDimensionArrayMember;


}

#endif //OSCDIMENSION_H
