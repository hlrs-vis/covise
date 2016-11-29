/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCLOGICS_H
#define OSCLOGICS_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscLogics : public oscObjectBase
{
public:
    oscLogics()
    {
        OSC_ADD_MEMBER(openDRIVE);
    };
    oscString openDRIVE;

};

typedef oscObjectVariable<oscLogics *> oscLogicsMember;


}

#endif //OSCLOGICS_H
