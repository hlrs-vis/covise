/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCAXLES_H
#define OSCAXLES_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscAxle.h"
#include "schema/oscAxle.h"
#include "schema/oscAxle.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscAxles : public oscObjectBase
{
public:
    oscAxles()
    {
        OSC_OBJECT_ADD_MEMBER(Front, "oscAxle");
        OSC_OBJECT_ADD_MEMBER(Rear, "oscAxle");
        OSC_OBJECT_ADD_MEMBER_OPTIONAL(Additional, "oscAxle");
    };
    oscAxleMember Front;
    oscAxleMember Rear;
    oscAxleMember Additional;

};

typedef oscObjectVariable<oscAxles *> oscAxlesMember;


}

#endif //OSCAXLES_H
