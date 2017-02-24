/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCNONE_H
#define OSCNONE_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscNone : public oscObjectBase
{
public:
oscNone()
{
    };
        const char *getScope(){return "";};

};

typedef oscObjectVariable<oscNone *> oscNoneMember;
typedef oscObjectVariableArray<oscNone *> oscNoneArrayMember;


}

#endif //OSCNONE_H
