/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl - 2.1.txt.

* License: LGPL 2 + */


#ifndef OSCFILEHEADER_H
#define OSCFILEHEADER_H

#include "../oscExport.h"
#include "../oscObjectBase.h"
#include "../oscObjectVariable.h"
#include "../oscObjectVariableArray.h"

#include "../oscVariables.h"

namespace OpenScenario
{
class OPENSCENARIOEXPORT oscFileHeader : public oscObjectBase
{
public:
oscFileHeader()
{
        OSC_ADD_MEMBER(revMajor, 0);
        OSC_ADD_MEMBER(revMinor, 0);
        OSC_ADD_MEMBER(date, 0);
        OSC_ADD_MEMBER(description, 0);
        OSC_ADD_MEMBER(author, 0);
    };
        const char *getScope(){return "";};
    oscUShort revMajor;
    oscUShort revMinor;
    oscDateTime date;
    oscString description;
    oscString author;

};

typedef oscObjectVariable<oscFileHeader *> oscFileHeaderMember;
typedef oscObjectVariableArray<oscFileHeader *> oscFileHeaderArrayMember;


}

#endif //OSCFILEHEADER_H
