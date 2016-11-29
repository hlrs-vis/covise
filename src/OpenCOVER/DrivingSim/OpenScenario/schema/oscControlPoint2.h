/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCCONTROLPOINT2_H
#define OSCCONTROLPOINT2_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscControlPoint2 : public oscObjectBase
{
public:
    oscControlPoint2()
    {
        OSC_ADD_MEMBER(status);
    };
    oscString status;

};

typedef oscObjectVariable<oscControlPoint2 *> oscControlPoint2Member;


}

#endif //OSCCONTROLPOINT2_H
