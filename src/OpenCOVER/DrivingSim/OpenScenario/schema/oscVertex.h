/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCVERTEX_H
#define OSCVERTEX_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscPosition.h"
#include "schema/oscShape.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscVertex : public oscObjectBase
{
public:
    oscVertex()
    {
        OSC_OBJECT_ADD_MEMBER(Position, "oscPosition");
        OSC_OBJECT_ADD_MEMBER(Shape, "oscShape");
    };
    oscPositionMember Position;
    oscShapeMember Shape;

};

typedef oscObjectVariable<oscVertex *> oscVertexMember;


}

#endif //OSCVERTEX_H
