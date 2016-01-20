/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_INIT_POSITION_H
#define OSC_INIT_POSITION_H

#include <oscExport.h>
#include <oscPosition.h>
#include <oscObjectVariable.h>


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscInitPosition: public oscPosition
{
public:
    oscInitPosition()
    {

    };
};

typedef oscObjectVariable<oscInitPosition *>oscInitPositionMember;

}

#endif //OSC_INIT_POSITION_H
