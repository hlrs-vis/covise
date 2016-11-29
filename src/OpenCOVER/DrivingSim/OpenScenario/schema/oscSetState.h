/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCSETSTATE_H
#define OSCSETSTATE_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscSetState : public oscObjectBase
{
public:
    oscSetState()
    {
        OSC_ADD_MEMBER(name);
        OSC_ADD_MEMBER(state);
    };
    oscString name;
    oscString state;

};

typedef oscObjectVariable<oscSetState *> oscSetStateMember;


}

#endif //OSCSETSTATE_H
