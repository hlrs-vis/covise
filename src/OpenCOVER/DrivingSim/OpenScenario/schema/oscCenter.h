/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCCENTER_H
#define OSCCENTER_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscCenter : public oscObjectBase
{
public:
    oscCenter()
    {
        OSC_ADD_MEMBER(x);
        OSC_ADD_MEMBER(y);
        OSC_ADD_MEMBER(z);
    };
    oscDouble x;
    oscDouble y;
    oscDouble z;

};

typedef oscObjectVariable<oscCenter *> oscCenterMember;


}

#endif //OSCCENTER_H
