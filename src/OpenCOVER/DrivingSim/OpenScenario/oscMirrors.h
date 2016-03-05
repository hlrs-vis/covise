/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_MIRRORS_H
#define OSC_MIRRORS_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariableArray.h"

#include "oscMirror.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscMirrors: public oscObjectBase
{
public:
    oscMirrors()
    {
        OSC_OBJECT_ADD_MEMBER(mirror, "oscMirror");
    };

    oscMirrorMember mirror;
};

typedef oscObjectVariableArray<oscMirrors *> oscMirrorsMemberArray;

}

#endif //OSC_MIRRORS_H
