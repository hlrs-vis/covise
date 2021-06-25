/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCSPLINE_H
#define OSCSPLINE_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"
#include "oscControlPoint1.h"
#include "oscControlPoint2.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscSpline : public oscObjectBase
{
public:
oscSpline()
{
        OSC_OBJECT_ADD_MEMBER(ControlPoint1, "oscControlPoint1", 0);
        OSC_OBJECT_ADD_MEMBER(ControlPoint2, "oscControlPoint2", 0);
    };
        const char *getScope(){return "/OSCTrajectory/Vertex/Shape";};
    oscControlPoint1Member ControlPoint1;
    oscControlPoint2Member ControlPoint2;

};

typedef oscObjectVariable<oscSpline *> oscSplineMember;
typedef oscObjectVariableArray<oscSpline *> oscSplineArrayMember;


}

#endif //OSCSPLINE_H
