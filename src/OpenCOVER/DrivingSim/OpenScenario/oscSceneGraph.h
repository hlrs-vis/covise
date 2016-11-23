/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_SCENEGRAPH_H
#define OSC_SCENEGRAPH_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscSceneGraph: public oscObjectBase
{
public:
    oscSceneGraph()
    {
		OSC_ADD_MEMBER(name);
    };

    oscString name;
};

typedef oscObjectVariable<oscSceneGraph *> oscSceneGraphMember;

}

#endif //OSC_SCENEGRAPH_H
