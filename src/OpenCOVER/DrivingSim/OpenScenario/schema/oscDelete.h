/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCDELETE_H
#define OSCDELETE_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscDelete : public oscObjectBase
{
public:
oscDelete()
{
    };
        const char *getScope(){return "";};

};

typedef oscObjectVariable<oscDelete *> oscDeleteMember;
typedef oscObjectVariableArray<oscDelete *> oscDeleteArrayMember;


}

#endif //OSCDELETE_H
