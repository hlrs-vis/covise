/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_SPLINE_H
#define OSC_SPLINE_H

#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>

#include <oscVariables.h>


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscSpline: public oscObjectBase
{
public:
    oscSpline()
    {
        OSC_ADD_MEMBER(controlPoint1);
        OSC_ADD_MEMBER(controlPoint2);
    };

    oscString controlPoint1;
    oscString controlPoint2;
};

typedef oscObjectVariable<oscSpline *> oscSplineMember;

}

#endif //OSC_SPLINE_H
