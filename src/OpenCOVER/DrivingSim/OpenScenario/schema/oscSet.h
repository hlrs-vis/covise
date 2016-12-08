/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCSET_H
#define OSCSET_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"
#include "oscObjectVariableArray.h"

#include "oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscSet : public oscObjectBase
{
public:
oscSet()
{
        OSC_ADD_MEMBER(value);
    };
    oscString value;

};

typedef oscObjectVariable<oscSet *> oscSetMember;
typedef oscObjectVariableArray<oscSet *> oscSetArrayMember;


}

#endif //OSCSET_H
