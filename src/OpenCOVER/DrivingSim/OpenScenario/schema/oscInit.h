/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCINIT_H
#define OSCINIT_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscActions.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscInit : public oscObjectBase
{
public:
    oscInit()
    {
        OSC_OBJECT_ADD_MEMBER(Actions, "oscActions");
    };
    oscActionsMember Actions;

};

typedef oscObjectVariable<oscInit *> oscInitMember;


}

#endif //OSCINIT_H
