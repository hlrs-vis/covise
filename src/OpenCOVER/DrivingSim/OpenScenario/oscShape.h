/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_SHAPE_H
#define OSC_SHAPE_H

#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>

#include <oscVariables.h>
#include <oscClothoid.h>
#include <oscSpline.h>


namespace OpenScenario {

class OPENSCENARIOEXPORT purposeType: public oscEnumType
{
public:
    static purposeType *instance(); 
private:
    purposeType();
    static purposeType *inst;
};

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscShape: public oscObjectBase
{
public:
    oscShape()
    {
        OSC_ADD_MEMBER(purpose);
        OSC_ADD_MEMBER(polyline);
        OSC_OBJECT_ADD_MEMBER(clothoid, "oscClothoid");
        OSC_OBJECT_ADD_MEMBER(spline, "oscSpline");

        purpose.enumType = purposeType::instance();
    };

    oscEnum purpose;
    oscString polyline;
    oscClothoidMember clothoid;
    oscSplineMember spline;

    enum purpose
    {
        steering,
        positioning,
    };
};

typedef oscObjectVariable<oscShape *> oscShapeMember;

}

#endif //OSC_SCHAPE_H
