/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef OSC_USER_DEFINED_COMMAND_H
#define OSC_USER_DEFINED_COMMAND_H

#include "oscExport.h"
#include "oscObjectBase.h"
#include "oscObjectVariable.h"

#include "oscVariables.h"


namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscUserDefinedCommand: public oscObjectBase
{
public:
    oscUserDefinedCommand()
    {
        OSC_ADD_MEMBER(command);
    };

    oscString command;
};

typedef oscObjectVariable<oscUserDefinedCommand *> oscUserDefinedCommandMember;

}

#endif //OSC_USER_DEFINED_COMMAND_H
