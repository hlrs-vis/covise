/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_ENTITY_ADD_H
#define OSC_ENTITY_ADD_H

#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>

#include <oscVariables.h>
#include <oscPosition.h>


namespace OpenScenario {

class OpenScenarioBase;

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscEntityAdd: public oscObjectBase
{
public:
    oscEntityAdd()
    {	
        OSC_ADD_MEMBER(name);
        OSC_OBJECT_ADD_MEMBER(position, "oscPosition");
    };

    oscString name;
    oscPositionMember position;
};

typedef oscObjectVariable<oscEntityAdd *> oscEntityAddMember;

}

#endif //OSC_ENTITY_ADD_H
