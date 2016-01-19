/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_ORIENTATION_H
#define OSC_ORIENTATION_H

#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>

#include <oscVariables.h>


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscOrientation: public oscObjectBase
{
public:
    oscOrientation()
    {
        OSC_ADD_MEMBER(h);
        OSC_ADD_MEMBER(p);
        OSC_ADD_MEMBER(r);
    };

    oscDouble h;
    oscDouble p;
    oscDouble r;
};

typedef oscObjectVariable<oscOrientation *> oscOrientationMember;

}

#endif //OSC_ORIENTATION_H
