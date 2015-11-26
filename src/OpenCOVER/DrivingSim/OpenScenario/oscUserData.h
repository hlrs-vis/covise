/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */
#ifndef OSC_USERDATA_H
#define OSC_USERDATA_H
#include <oscExport.h>
#include <oscObjectBase.h>
#include <oscObjectVariable.h>
#include <oscVariables.h>

namespace OpenScenario {

/// \class This class represents a generic OpenScenario Object
class OPENSCENARIOEXPORT oscUserData : public oscObjectBase
{
public:
    oscUserData()
    {
        OSC_ADD_MEMBER(code);
        OSC_ADD_MEMBER(value);
    };
    oscString code;
    oscString value;
};

typedef oscObjectVariable<oscUserData *> oscUserDataMember;

}

#endif //OSC_USERDATA_H
