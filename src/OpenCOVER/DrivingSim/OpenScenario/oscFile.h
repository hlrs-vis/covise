/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_FILE_H
#define OSC_FILE_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>

namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscFile: public oscObjectBase
{
public:
    oscFile()
    {
        OSC_ADD_MEMBER(URL);
    };
    oscString URL;
};

typedef oscObjectVariable<oscFile *> oscFileMember;

}

#endif //OSC_FILE_H
