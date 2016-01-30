/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_CONTROL_POINT_H
#define OSC_CONTROL_POINT_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"


namespace OpenScenario {

class OPENSCENARIOEXPORT statusType: public oscEnumType
{
public:
    static statusType *instance();
private:
    statusType();
    static statusType *inst;
};

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscControlPoint: public oscObjectBase
{
public:
    oscControlPoint()
    {
        OSC_ADD_MEMBER(status);

        status.enumType = statusType::instance();
    };

    oscEnum status;

    enum status
    {
        toBeDefined
    };
};

typedef oscObjectVariable<oscControlPoint *> oscControlPointMember;

}

#endif /* OSC_CONTROL_POINT_H */
