/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_COMMAND_H
#define OSC_COMMAND_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>

namespace OpenScenario {


/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscCommand: public oscObjectBase
{
public:
    oscCommand()
    {
        OSC_ADD_MEMBER(name);
    };
    oscString name;
};

typedef oscObjectVariable<oscCommand *> oscCommandMember;

}

#endif //OSC_COMMAND_H