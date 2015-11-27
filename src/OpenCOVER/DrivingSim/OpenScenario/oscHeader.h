/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_HEADER_H
#define OSC_HEADER_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>

namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscHeader: public oscObjectBase
{
public:
    oscHeader()
    {
        OSC_ADD_MEMBER(revMajor);
        OSC_ADD_MEMBER(revMinor);
        OSC_ADD_MEMBER(description);
        OSC_ADD_MEMBER(date);
        OSC_ADD_MEMBER(author);
    }
    oscShort revMajor;
    oscShort revMinor;
    oscString description;
    oscString date;
    oscString author;
};

typedef oscObjectVariable<oscHeader *> oscHeaderMember;

}

#endif //OSC_HEADER_H
