/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_TIME_H
#define OSC_TIME_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscTime: public oscObjectBase
{
public:
    oscTime()
    {
        OSC_ADD_MEMBER(hour);
        OSC_ADD_MEMBER(min);
        OSC_ADD_MEMBER(sec);
    };

    oscUInt hour;
    oscUInt min;
    oscDouble sec;
};

typedef oscObjectVariable<oscTime *> oscTimeMember;

}

#endif //OSC_TIME_H
