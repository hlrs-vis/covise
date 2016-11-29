/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCLONGITUDINAL_H
#define OSCLONGITUDINAL_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscEmpty.h"
#include "schema/oscTiming.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscLongitudinal : public oscObjectBase
{
public:
    oscLongitudinal()
    {
        OSC_OBJECT_ADD_MEMBER(None, "oscEmpty");
        OSC_OBJECT_ADD_MEMBER(Timing, "oscTiming");
    };
    oscEmptyMember None;
    oscTimingMember Timing;

};

typedef oscObjectVariable<oscLongitudinal *> oscLongitudinalMember;


}

#endif //OSCLONGITUDINAL_H
