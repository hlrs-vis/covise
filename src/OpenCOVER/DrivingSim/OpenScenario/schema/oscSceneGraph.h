/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCSCENEGRAPH_H
#define OSCSCENEGRAPH_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscSceneGraph : public oscObjectBase
{
public:
oscSceneGraph()
{
        OSC_ADD_MEMBER(name, 0);
    };
    oscString name;

};

typedef oscObjectVariable<oscSceneGraph *> oscSceneGraphMember;
typedef oscObjectVariableArray<oscSceneGraph *> oscSceneGraphArrayMember;


}

#endif //OSCSCENEGRAPH_H
