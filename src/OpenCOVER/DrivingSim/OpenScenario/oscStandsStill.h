/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_STANDS_STILL_H
#define OSC_STANDS_STILL_H

#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>

#include <oscVariables.h>


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscStandsStill: public oscObjectBase
{
public:
    oscStandsStill()
    {
        OSC_ADD_MEMBER(object);
        OSC_ADD_MEMBER(duration);
    };

    oscString object;
    oscDouble duration;
};

typedef oscObjectVariable<oscStandsStill *> oscStandsStillMember;

}

#endif //OSC_STANDS_STILL_H
