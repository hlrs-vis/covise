/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_ROAD_CONDITIONS_H
#define OSC_ROAD_CONDITIONS_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectArrayVariable.h"

#include "oscRoadCondition.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscRoadConditions: public oscObjectBase
{
public:
    oscRoadConditions()
    {
        OSC_OBJECT_ADD_MEMBER(roadCondition, "oscRoadCondition");
    };

	oscRoadConditionMember roadCondition;
};

typedef oscObjectArrayVariable<oscRoadConditions *> oscRoadConditionsArrayMember;

}

#endif //OSC_ROAD_CONDITIONS_H
