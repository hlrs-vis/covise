/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_ORIENTATION_OPTIONAL_H
#define OSC_ORIENTATION_OPTIONAL_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscOrientationOptional: public oscObjectBase
{
public:
    oscOrientationOptional()
    {
        OSC_ADD_MEMBER_OPTIONAL(h);
        OSC_ADD_MEMBER_OPTIONAL(p);
        OSC_ADD_MEMBER_OPTIONAL(r);
    };

    oscDouble h;
    oscDouble p;
    oscDouble r;
};

typedef oscObjectVariable<oscOrientationOptional *> oscOrientationOptionalMember;

}

#endif /* OSC_ORIENTATION_OPTIONAL_H */
