/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCEMPTY_H
#define OSCEMPTY_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscEmpty : public oscObjectBase
{
public:
    oscEmpty()
    {
    };

};

typedef oscObjectVariable<oscEmpty *> oscEmptyMember;


}

#endif //OSCEMPTY_H
