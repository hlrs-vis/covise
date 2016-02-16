/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_EYEPOINTS_H
#define OSC_EYEPOINTS_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariableArray.h"

#include "oscEyepoint.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscEyepoints : public oscObjectBase
{
public:
    oscEyepoints()
    {
        OSC_OBJECT_ADD_MEMBER(eyepoint, "oscEyepoint");
    };

    oscEyepointMember eyepoint;
};

typedef oscObjectVariableArray<oscEyepoints *> oscEyepointsMemberArray;

}

#endif //OSC_EYEPOINTS_H
