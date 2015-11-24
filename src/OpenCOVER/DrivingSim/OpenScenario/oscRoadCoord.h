/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_ROAD_COORD_H
#define OSC_ROAD_COORD_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>

namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscRoadCoord: public oscObjectBase
{
public:
    oscRoadCoord()
    {
        OSC_ADD_MEMBER(pathS);
        OSC_ADD_MEMBER(t);
    };
    oscDouble pathS;
    oscDouble t;
};

typedef oscObjectVariable<oscRoadCoord *> oscRoadCoordMember;

}

#endif //OSC_ROAD_COORD_H
