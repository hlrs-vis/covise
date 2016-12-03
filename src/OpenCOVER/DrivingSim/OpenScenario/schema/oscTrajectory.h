/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCTRAJECTORY_H
#define OSCTRAJECTORY_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"
#include "schema/oscExtent.h"
#include "schema/oscVertex.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscTrajectory : public oscObjectBase
{
public:
    oscTrajectory()
    {
        OSC_ADD_MEMBER(name);
        OSC_ADD_MEMBER(closed);
        OSC_ADD_MEMBER(domain);
        OSC_OBJECT_ADD_MEMBER(Vertex, "oscVertex");
    };
    oscString name;
    oscBool closed;
    oscEnum domain;
    oscVertexMember Vertex;

};

typedef oscObjectVariable<oscTrajectory *> oscTrajectoryMember;


}

#endif //OSCTRAJECTORY_H
