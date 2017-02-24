/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCFILE_H
#define OSCFILE_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscFile : public oscObjectBase
{
public:
oscFile()
{
        OSC_ADD_MEMBER(filepath, 0);
    };
        const char *getScope(){return "";};
    oscString filepath;

};

typedef oscObjectVariable<oscFile *> oscFileMember;
typedef oscObjectVariableArray<oscFile *> oscFileArrayMember;


}

#endif //OSCFILE_H
