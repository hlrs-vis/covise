/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCSHAPE_H
#define OSCSHAPE_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscPolyline.h"
#include "oscClothoid.h"
#include "oscSpline.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscShape : public oscObjectBase
{
public:
oscShape()
{
        OSC_ADD_MEMBER(reference, 0);
        OSC_OBJECT_ADD_MEMBER(Polyline, "oscPolyline", 1);
        OSC_OBJECT_ADD_MEMBER(Clothoid, "oscClothoid", 1);
        OSC_OBJECT_ADD_MEMBER(Spline, "oscSpline", 1);
    };
        const char *getScope(){return "/OSCTrajectory/Vertex";};
    oscDouble reference;
    oscPolylineMember Polyline;
    oscClothoidMember Clothoid;
    oscSplineMember Spline;

};

typedef oscObjectVariable<oscShape *> oscShapeMember;
typedef oscObjectVariableArray<oscShape *> oscShapeArrayMember;


}

#endif //OSCSHAPE_H
