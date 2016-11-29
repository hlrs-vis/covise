/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCPRIVATE_H
#define OSCPRIVATE_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscPrivateAction.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscPrivate : public oscObjectBase
{
public:
    oscPrivate()
    {
        OSC_ADD_MEMBER(object);
        OSC_OBJECT_ADD_MEMBER(Action, "oscPrivateAction");
    };
    oscString object;
    oscPrivateActionMember Action;

};

typedef oscObjectVariable<oscPrivate *> oscPrivateMember;


}

#endif //OSCPRIVATE_H
