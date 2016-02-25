/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_LATERAL_H
#define OSC_LATERAL_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"


namespace OpenScenario {

class OPENSCENARIOEXPORT purposeLateralType: public oscEnumType
{
public:
    static purposeLateralType *instance();
private:
    purposeLateralType();
    static purposeLateralType *inst;
};

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscLateral: public oscObjectBase
{
public:
    oscLateral()
    {
        OSC_ADD_MEMBER(purpose);

        purpose.enumType = purposeLateralType::instance();
    };

    oscEnum purpose;

    enum purposeLateral
    {
        position,
        steering,
        navigation
    };
};

typedef oscObjectVariable<oscLateral *> oscLateralMember;

}

#endif /* OSC_LATERAL_H */
