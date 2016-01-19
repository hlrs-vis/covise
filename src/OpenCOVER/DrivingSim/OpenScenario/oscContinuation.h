/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_CONTINUATION_H
#define OSC_CONTINUATION_H

#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>

#include <oscRoute.h>
#include <oscShape.h>


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscContinuation: public oscObjectBase
{
public:
    oscContinuation()
    {
        OSC_OBJECT_ADD_MEMBER(route, "oscRoute");
        OSC_OBJECT_ADD_MEMBER(shape, "oscShape");
    };

    oscRouteMember route;
    oscShapeMember shape;
};

typedef oscObjectVariable<oscContinuation *> oscContinuationMember;

}

#endif //OSC_CONTINUATION_H
