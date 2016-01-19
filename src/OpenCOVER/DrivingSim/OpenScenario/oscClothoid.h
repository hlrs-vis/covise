/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_CLOTHOID_H
#define OSC_CLOTHOID_H

#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>

#include <oscVariables.h>


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscClothoid: public oscObjectBase
{
public:
    oscClothoid()
    {
        OSC_ADD_MEMBER(curvature);
        OSC_ADD_MEMBER(curvatureDot);
        OSC_ADD_MEMBER(length);
    };

    oscDouble curvature;
    oscDouble curvatureDot;
    oscDouble length;
};

typedef oscObjectVariable<oscClothoid *> oscClothoidMember;

}

#endif //OSC_CLOTHOID_H
