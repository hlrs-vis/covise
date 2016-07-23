/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_ROAD_CONDITION_GROUP_H
#define OSC_ROAD_CONDITION_GROUP_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "oscRoadConditions.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscRoadConditionsGroup: public oscObjectBase
{
public:
    oscRoadConditionsGroup()
    {
        OSC_ADD_MEMBER(frictionScale);
        OSC_OBJECT_ADD_MEMBER(roadConditions, "oscRoadConditions");
    };

    oscDouble frictionScale;
    oscRoadConditionsArrayMember roadConditions;
};

typedef oscObjectVariable<oscRoadConditionsGroup *> oscRoadConditionsGroupMember;

}

#endif /* OSC_ROAD_CONDITION_GROUP_H */
