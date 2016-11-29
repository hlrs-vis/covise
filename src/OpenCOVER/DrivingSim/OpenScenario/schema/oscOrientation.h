/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCORIENTATION_H
#define OSCORIENTATION_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscOrientation.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscOrientation : public oscObjectBase
{
public:
    oscOrientation()
    {
        OSC_ADD_MEMBER_OPTIONAL(type);
        OSC_ADD_MEMBER_OPTIONAL(h);
        OSC_ADD_MEMBER_OPTIONAL(p);
        OSC_ADD_MEMBER_OPTIONAL(r);
    };
    oscEnum type;
    oscDouble h;
    oscDouble p;
    oscDouble r;

};

typedef oscObjectVariable<oscOrientation *> oscOrientationMember;


}

#endif //OSCORIENTATION_H
