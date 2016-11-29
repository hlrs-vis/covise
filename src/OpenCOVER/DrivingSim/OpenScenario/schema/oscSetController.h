/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCSETCONTROLLER_H
#define OSCSETCONTROLLER_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscSetController : public oscObjectBase
{
public:
    oscSetController()
    {
        OSC_ADD_MEMBER(name);
        OSC_ADD_MEMBER(state);
    };
    oscString name;
    oscString state;

};

typedef oscObjectVariable<oscSetController *> oscSetControllerMember;


}

#endif //OSCSETCONTROLLER_H
