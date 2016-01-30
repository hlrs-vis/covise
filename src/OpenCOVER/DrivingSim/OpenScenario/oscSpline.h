/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_SPLINE_H
#define OSC_SPLINE_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscControlPoint.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscSpline: public oscObjectBase
{
public:
    oscSpline()
    {
        OSC_OBJECT_ADD_MEMBER(controlPoint1, "oscControlPoint");
        OSC_OBJECT_ADD_MEMBER(controlPoint2, "oscControlPoint");
    };

    oscControlPointMember controlPoint1;
    oscControlPointMember controlPoint2;
};

typedef oscObjectVariable<oscSpline *> oscSplineMember;

}

#endif //OSC_SPLINE_H
