/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_FRUSTUM_H
#define OSC_FRUSTUM_H

#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>

#include <oscVariables.h>


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscFrustum: public oscObjectBase
{
public:
    oscFrustum()
    {
        OSC_ADD_MEMBER(near);
        OSC_ADD_MEMBER(far);
        OSC_ADD_MEMBER(left);
        OSC_ADD_MEMBER(right);
        OSC_ADD_MEMBER(bottom);
        OSC_ADD_MEMBER(top);
    };

    oscDouble near;
    oscDouble far;
    oscDouble left;
    oscDouble right;
    oscDouble bottom;
    oscDouble top;
};

typedef oscObjectVariable<oscFrustum *> oscFrustumMember;

}

#endif //OSC_FRUSTUM_H
