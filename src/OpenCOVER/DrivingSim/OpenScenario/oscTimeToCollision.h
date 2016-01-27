/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_TIME_TO_COLLISION_H
#define OSC_TIME_TO_COLLISION_H

#include "oscExport.h"
#include "oscConditionChoiceBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "oscPosition.h"


namespace OpenScenario {


/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscTimeToCollision: public oscConditionChoiceBase
{
public:
    oscTimeToCollision()
    {	
        OSC_ADD_MEMBER(freespace);
        OSC_OBJECT_ADD_MEMBER(position, "oscPosition");
    };

    oscBool freespace;
    oscPositionMember position;
};

typedef oscObjectVariable<oscTimeToCollision *> oscTimeToCollisionMember;

}

#endif //OSC_TIME_TO_COLLISION_H
