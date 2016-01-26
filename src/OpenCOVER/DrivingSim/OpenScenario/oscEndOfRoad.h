/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_END_OF_ROAD_H
#define OSC_END_OF_ROAD_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscEndOfRoad: public oscObjectBase
{
public:
    oscEndOfRoad()
    {

    };
};

typedef oscObjectVariable<oscEndOfRoad *> oscEndOfRoadMember;

}


#endif /* OSC_END_OF_ROAD_H */
